import wandb


class WandBLogger():
	def __init__(self, key, proj_name, name, cfg):
		wandb.login(key=key)
		self.run = wandb.init(project=proj_name, name=name, config=cfg)

	def log(self, name, value):
		wandb.log({name, value})

	def kill(self):
		wandb.finish()
