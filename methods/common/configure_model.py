from importlib import import_module

def get_h2o_args(parser):
    parser.add_argument("--use-h2o", action='store_true', default=False, help="use the H2O algos")
    parser.add_argument("--heavy-ratio", type=float, default=0.1, help="H2O heavy ratio," "set to 0.1 by default")
    return parser

def get_energy_args(parser):
    parser.add_argument("--use-energy-evict", action='store_true', default=False,
                        help="use the query-weighted energy KV evictor")
    parser.add_argument("--keep-ratio", type=float, default=0.1,
                        help="fraction of tokens kept by energy top-k")
    if not any("--recent-ratio" in action.option_strings for action in parser._actions):
        parser.add_argument("--recent-ratio", type=float, default=0.1,
                            help="fraction of tokens kept by the recent window")
    return parser

def _add_arg_if_missing(parser, option, *args, **kwargs):
    if not any(option in action.option_strings for action in parser._actions):
        parser.add_argument(option, *args, **kwargs)

def get_eviction_args(parser):
    _add_arg_if_missing(parser, "--use-evict", action="store_true", default=False)
    _add_arg_if_missing(parser, "--evict-method", type=str, default="value_residual",
                        choices=["full", "snapkv", "h2o", "value_residual"])
    _add_arg_if_missing(parser, "--evict-budget", type=float, default=512,
                        help="<1 = fraction of prompt kept; >=1 = absolute token count")
    _add_arg_if_missing(parser, "--obs-window", type=int, default=32,
                        help="last N prompt tokens used as the query window")
    _add_arg_if_missing(parser, "--sink-tokens", type=int, default=4)
    _add_arg_if_missing(parser, "--recent-tokens", type=int, default=64)
    _add_arg_if_missing(parser, "--pool-kernel", type=int, default=7)
    _add_arg_if_missing(parser, "--cand-mult", type=int, default=3)
    return parser

def get_topk_args(parser):
    parser.add_argument("--use-topk", action='store_true', default=False, help="use the H2O algos")
    parser.add_argument("--top-k", type=float, default=-1, help="top k tokens to consider - >1 exact number, <1 fraction of tokens to consider," "set to -1 to use all tokens")
    return parser

def get_pca_args(parser):
    parser.add_argument("--top-r", type=float, default=-1, help="top r channels to consider," "set to -1 to use all channels")
    parser.add_argument("--use-pca-topk", action='store_true', default=False, help="use the PCA TopK algos")
    parser.add_argument("--rotary-type", type=str, default="postrotary", help="rotary type")
    parser.add_argument("--recent-ratio", type=float, default=-1, help="PcaTopK recent ratio," "set to -1 by default")
    parser.add_argument("--transform-dataset", type=str, default="wikitext", help="pca transform dataset - wikitext, bookcorpus, c4")
    return parser

def get_save_tensor_args(parser):
    parser.add_argument("--save-tensors", action='store_true', default=False, help="save tensor to file")
    parser.add_argument("--tensors-dir", type=str, default="./", help="file to save tensor to")
    return parser


def get_modifier(args):
    if args.use_topk:
        method_name = "top_k"
        module_name = ".baselines.topk.modify_" + args.model_type
    elif args.use_h2o:
        method_name = "h2o"
        module_name = ".baselines.h2o.modify_" + args.model_type
    elif args.use_energy_evict:
        method_name = "energy"
        module_name = ".baselines.energy.modify_" + args.model_type
    elif getattr(args, "use_evict", False):
        method_name = "eviction"
        module_name = ".eviction.modify_" + args.model_type
    elif args.use_pca_topk:
        method_name = "pca_topk"
        module_name = ".pca_topk.modify_" + args.model_type
    else:
        return None
      
    method_name = "make_" + args.model_type + "_attention_" + method_name
    module = import_module(module_name, package="methods")
    method = getattr(module, method_name)
    return method

def get_config_dict(args):
    config_dict = {}
    config_dict["model"] = args.model_id
    config_dict["sequence_length"] = args.sequence_length
    if args.use_h2o:
        config_dict["method"] = "h2o"
        config_dict["heavy_ratio"] = args.heavy_ratio
        if args.recent_ratio != -1:
            print ("[WARNING] Recent Ratio is override by Heavy Ratio for H2O")
        config_dict["recent_ratio"] = args.heavy_ratio
    elif args.use_energy_evict:
        config_dict["method"] = "energy"
        config_dict["keep_ratio"] = args.keep_ratio
        config_dict["recent_ratio"] = args.recent_ratio if args.recent_ratio != -1 else args.keep_ratio
    elif getattr(args, "use_evict", False):
        config_dict["method"] = "eviction"
        config_dict["evict_method"] = args.evict_method
        config_dict["evict_budget"] = args.evict_budget
        config_dict["sink_tokens"] = args.sink_tokens
        config_dict["recent_tokens"] = args.recent_tokens
        config_dict["obs_window"] = args.obs_window
    elif args.use_topk:
        config_dict["method"] = "topk"
        config_dict["top_k"] = args.top_k
    elif args.use_pca_topk:
        config_dict["method"] = "pca_topk"
        config_dict["top_r"] = args.top_r
        config_dict["top_k"] = args.top_k
        config_dict["rotary_type"] = args.rotary_type
        config_dict["recent_ratio"] = args.recent_ratio
        config_dict["transform_dataset"] = args.transform_dataset
    else:
        config_dict["method"] = "base_hf"
    return config_dict
