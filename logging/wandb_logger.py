import wandb


class WandBLogger():
	def __init__(self, key, proj_name, name, cfg, run_id=None):
		wandb.login(key=key)
		if run_id is None:
			self.run = wandb.init(project=proj_name, name=name, config=cfg)
		else:
			self.run = wandb.init(project=proj_name, name=name, config=cfg, id=run_id, resume='must')

	def log(self, name, value):
		wandb.log({name : value})

	def kill(self):
		wandb.finish()
