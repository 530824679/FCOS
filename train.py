
import os
import torch
import argparse

def train(cfg):
    device = torch.device()

    model = build_model(cfg).to(device)

    optimizer = make_optimizer(cfg, model)

    scheduler = make_lr_scheduler(cfg, optimizer)

    dataset = make_dataloader(cfg, is_train=True)

    for iter, (images, targets, _) in enumerate(dataset, start_iter):
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        scheduler.step()

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)



def main():
    parser = argparse.ArgumentParser(description="FCOS Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--skip-test", dest="skip_test", help="Do not test the final model", action="store_true")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))

