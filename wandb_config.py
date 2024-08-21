'''
Weights and Biases config to avoid code in main scripts
'''
import wandb

def setup_wandb():
    #logging info
    #Set an experiment name to group training and evaluation
    experiment_id = wandb.util.generate_id()

    # Start a run, tracking hyperparameters
    wandb.init(
        project="ImageNet",
        group="GFNet",
        config={
            "id": experiment_id,
            #~ "machine": "Xeon-3090-RTX",
            "machine": "H100",
            "architecture": "gfnet-xs",
            "model": "GFNet",
            "dataset": "ImageNet",
            "epochs": 300,
            "optimizer": "adam",
            "loss": "crossentropy",
            "metric": "accuracy",
            #~ "dim": 64,
            "depth": 12,
            "embed_dim": 384,
            "batch_size": 128
        })
    config = wandb.config
