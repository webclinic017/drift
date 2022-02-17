import wandb
import os


def get_wandb():
    from dotenv import load_dotenv

    load_dotenv()

    """ 0. Login to Weights and Biases """
    wsb_token = os.environ.get("WANDB_API_KEY")
    if wsb_token:
        wandb.login(key=wsb_token)
        return wandb
    else:
        return None  # wandb.login()

    # return wandb
