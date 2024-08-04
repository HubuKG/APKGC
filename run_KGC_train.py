from email.generator import Generator
import torch
import mmkgc
from mmkgc.config import Trainer, Tester
from mmkgc.module.model import APKGC
from mmkgc.module.loss import MarginLoss, SigmoidLoss
from mmkgc.module.strategy import NegativeSampling
from mmkgc.data import TrainDataLoader, TestDataLoader
from torchlight import initialize_exp, get_dump_path
from args import get_args
import os.path as osp
import os
import pdb

if __name__ == "__main__":
    args = get_args()
    this_dir = osp.dirname(__file__)
    data_root = osp.abspath(osp.join(this_dir, '..', '..', 'data', ''))
    data_path = osp.join(data_root, args.data_path)
    args.dump_path = osp.join(data_path, args.dump_path)
    save_path = osp.join(data_path, "checkpoint")
    
    if not osp.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    args.save = osp.join(save_path, args.save)
    args.exp_name = f"{args.exp_name}-{args.dataset}"

    logger = initialize_exp(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )
    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")
    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')
    kge_score = APKGC(
        args=args,
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        margin=args.margin,
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb
    )
    logger.info(kge_score)
    model = NegativeSampling(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
    )

    trainer = Trainer(
        args=args,
        logger=logger,
        model=model,
        data_loader=train_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        use_gpu=True,
        opt_method='Adam',
        train_mode='normal',
        save_steps=100,
        checkpoint_dir=args.save
    )

    trainer.run()

    save_dir = f"{args.save}-MRR{trainer.Loss_log.get_acc()}"
    if not osp.exists(save_dir):
        torch.save(trainer.best_model_wts, f"{args.save}-MRR{trainer.Loss_log.get_acc()}")

    kge_score.load_checkpoint(save_dir)
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)
    logger.info(f"mrr:{mrr},\t mr:{mr},\t hit10:{hit10},\t hit3:{hit3},\t hit1:{hit1}")
    logger.info(f"{mrr}\t{mr}\t{hit10}\t{hit3}\t{hit1}")
    logger.info(" -------------------- finish! -------------------- ")