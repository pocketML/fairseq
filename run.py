import torch
from fairseq import distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_cli.train import main
from pocketml.args import create_parser, parse_roberta_args

if __name__ == "__main__":
    parser = create_parser()

    roberta_args = parse_roberta_args(parser)

    cfg = convert_namespace_to_omegaconf(roberta_args)

    if roberta_args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)
