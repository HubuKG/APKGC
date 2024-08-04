import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K')
    arg.add_argument('-batch_size', type=int, default=1024)
    arg.add_argument('-margin', type=float, default=6.0)
    arg.add_argument('-dim', type=int, default=128)
    arg.add_argument('-epoch', type=int, default=1000)
    arg.add_argument('-save', type=str)
    arg.add_argument('-img_dim', type=int, default=4096)
    arg.add_argument('-neg_num', type=int, default=1)
    arg.add_argument('-learning_rate', type=float, default=0.001)
    arg.add_argument('-lrg', type=float, default=0.001)
    arg.add_argument('-adv_temp', type=float, default=2.0)
    arg.add_argument('-visual', type=str, default='random')
    arg.add_argument('-seed', type=int, default=42)
    arg.add_argument('-missing_rate', type=float, default=0.8)
    arg.add_argument('-postfix', type=str, default='')
    arg.add_argument('-con_temp', type=float, default=0)
    arg.add_argument('-early_stop', type=int, default=10)
    arg.add_argument('-lamda', type=float, default=0)
    arg.add_argument('-mu', type=float, default=0)

    arg.add_argument("-no_tensorboard", default=False, action="store_true")
    arg.add_argument("-exp_name", default="KGC_exp", type=str, help="Experiment name")
    arg.add_argument("-dump_path", default="dump/", type=str, help="Experiment dump path")
    arg.add_argument("-exp_id", default="001", type=str, help="Experiment ID")
    arg.add_argument("-data_path", default="mmkgc", type=str, help="Experiment path")
    
    arg.add_argument('-num_proj', type=int, default=1)

    arg.add_argument('-add_noise', type=int, default=0, choices=[0, 1])
    arg.add_argument('-use_pool', type=int, default=1, choices=[0, 1])

    arg.add_argument('-noise_update', type=str, default="epoch", choices=["epoch", "step"])
    arg.add_argument('-noise_ratio', type=float, default=0.1)
    arg.add_argument('-mask_ratio', type=float, default=0.1)
    arg.add_argument('-joint_way', type=str, default="Mformer_hd_mean", choices=["Mformer_hd_mean", "Mformer_hd_graph", "Mformer_weight", "atten_weight", "learnable_weight", "cat_mlp"])

    arg.add_argument("-hidden_size", type=int, default=256, help="the hidden size of MEAformer")
    arg.add_argument("-intermediate_size", type=int, default=256, help="the hidden size of MEAformer")
    arg.add_argument("-num_attention_heads", type=int, default=1, help="the number of attention_heads of MEAformer")
    arg.add_argument("-num_hidden_layers", type=int, default=1, help="the number of hidden_layers")
    arg.add_argument("-use_intermediate", type=int, default=0, help="whether to use_intermediate")
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
