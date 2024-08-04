import torch
import mmkgc
from mmkgc.config import Trainer, Tester
from mmkgc.module.model import APKGC
from mmkgc.data import TrainDataLoader, TestDataLoader
from args import get_args
import os.path as osp

if __name__ == "__main__":
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")
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
    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')
    kge_score = AdvMixRotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        margin=args.margin,
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb
    )
    print(kge_score)



    save_dir = ""
    kge_score.load_checkpoint(save_dir)
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
