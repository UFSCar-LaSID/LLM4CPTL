# --- INÍCIO DA NOSSA IMPLEMENTAÇÃO (MÊS 1) ---
import torch
import random


class EmbeddingHistoryBuffer:
    def __init__(self):
        self.history_embeddings = []
        self.history_counts = []
        self.history_mu = []
        self.history_logvar = []

        self.current_task_embeddings = []
        self.current_task_counts = []

    def accumulate_current_task(self, batch_embeddings):
        batch_embs_cpu = batch_embeddings.detach().cpu()
        for emb in batch_embs_cpu:
            found = False
            for i, saved_emb in enumerate(self.current_task_embeddings):
                if torch.allclose(emb, saved_emb, atol=1e-5):
                    self.current_task_counts[i] += 1
                    found = True
                    break
            if not found:
                self.current_task_embeddings.append(emb)
                self.current_task_counts.append(1)

    def commit_task(self, task_mu=None, task_logvar=None):
        self.history_embeddings.extend(self.current_task_embeddings)
        self.history_counts.extend(self.current_task_counts)

        # Guarda a mesma âncora latente global para todos os cenários dessa tarefa
        for _ in range(len(self.current_task_embeddings)):
            self.history_mu.append(task_mu)
            self.history_logvar.append(task_logvar)

        self.current_task_embeddings = []
        self.current_task_counts = []

    def sample_embeddings(self, batch_size, device):
        if not self.history_embeddings:
            raise ValueError("Buffer de histórico vazio.")

        sampled_indices = random.choices(
            range(len(self.history_embeddings)),
            weights=self.history_counts,
            k=batch_size
        )

        sampled_embs = [self.history_embeddings[idx].unsqueeze(
            0) for idx in sampled_indices]

        # Recupera os priors se existirem
        sampled_mus = [self.history_mu[idx]
                       for idx in sampled_indices if self.history_mu[idx] is not None]
        sampled_logvars = [self.history_logvar[idx]
                           for idx in sampled_indices if self.history_logvar[idx] is not None]

        embs_tensor = torch.cat(sampled_embs, dim=0).to(device)
        mus_tensor = torch.cat(sampled_mus, dim=0).to(
            device) if sampled_mus else None
        logvars_tensor = torch.cat(sampled_logvars, dim=0).to(
            device) if sampled_logvars else None

        return embs_tensor, mus_tensor, logvars_tensor
