import wandb
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from config import cfg
from dataset import *

# Initialize W&B logging
wandb.init(project="maskrcnn-kartai", config=cfg)

# Custom trainer that logs training metrics to wandb
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = "./output"
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)

    def run_step(self):
        super().run_step()
        metrics = self.storage.latest()

        def get_scalar(x):
            return float(x[0]) if isinstance(x, tuple) else float(x)

        wandb.log({
            "iter": float(self.iter),
            "total_loss": get_scalar(metrics.get("total_loss", 0)),
            "loss_cls": get_scalar(metrics.get("loss_cls", 0)),
            "loss_box_reg": get_scalar(metrics.get("loss_box_reg", 0)),
            "loss_mask": get_scalar(metrics.get("loss_mask", 0)),
            "loss_rpn_cls": get_scalar(metrics.get("loss_rpn_cls", 0)),
            "loss_rpn_loc": get_scalar(metrics.get("loss_rpn_loc", 0)),
        })


if __name__ == "__main__":
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Run evaluation
    evaluator = COCOEvaluator("building_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "building_dataset_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)

    print("\n📊 Evaluation Results:")
    print(results)

    # Log evaluation results to wandb (flattened)
    for eval_type, metrics in results.items():
        for k, v in metrics.items():
            wandb.log({f"{eval_type}/{k}": float(v)})

    # Finish wandb session
    wandb.finish()
