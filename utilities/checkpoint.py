import torch
from os.path import isdir, join
from os import mkdir


class PRENCheckpoint:
    def __init__(self, workspace: str, resume: str):
        workspace = join(workspace, 'checkpoint')
        if not isdir(workspace):
            mkdir(workspace)
        workspace = join(workspace, 'recognizer')
        if not isdir(workspace):
            mkdir(workspace)
        self.workspace: str = workspace
        self.resume: str = resume

    def save(self, model, optimizer, step: int):
        save_path: str = join(self.workspace, 'checkpoint{}.pth'.format(step))
        torch.save({
            'model': model.state_dict(),
        }, save_path)
        last_path: str = join(self.workspace, 'last.pth')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step
        }, last_path)

    def load(self):
        if (self.resume is None) or len(self.resume) == 0:
            return None
        state_dict: dict = torch.load(self.resume)
        return state_dict

    def load_path(self, path: str, device):
        state_dict: dict = torch.load(path, map_location=device)
        return state_dict
