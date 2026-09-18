"""
connectome_rnn.py
===================
FLYNN 논문의 핵심 트릭을 그대로 구현한다: 뇌의 "연결 구조(토폴로지)"는
커넥톰에서 고정으로 가져오고, 그 위의 "가중치 크기"만 RL로 학습한다.

leaky RNN 업데이트 규칙 (연속시간 근사):
    h_{t+1} = h_t + (dt/tau) * ( -h_t + f(A_signed * scale * h_t + W_in @ x_t) )
    y_t     = W_out @ h_t[output_idx]

- A_signed: 커넥톰에서 온 고정 토폴로지(부호 포함, 0인 곳은 연결 없음 — 학습으로도 안 바뀜)
- scale   : A_signed와 같은 shape의 학습 가능한 스칼라 배율 마스크
            (토폴로지는 고정, "세기"만 배우게 하는 부분)
- W_in/W_out: 입력/출력 인터페이스 (일반 학습 가능 선형층)

두 버전 제공:
- ConnectomeRNN (torch.nn.Module): 실제 학습용. torch 필요.
- reference_forward_numpy(): 위 수식을 numpy로 그대로 구현한 참조 구현.
  torch 없이도 로직이 맞는지 검증할 수 있게 별도로 뒀다 — 이 개발 환경에는
  torch가 없어서, 아래 두 구현이 같은 결과를 내는지는 사용자 환경에서
  test_equivalence()로 직접 재확인 필요.
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = None


# ---------------------------------------------------------------------- #
# numpy 참조 구현 (torch 없이도 로직 검증 가능)
# ---------------------------------------------------------------------- #
def reference_forward_numpy(A_signed: np.ndarray, scale: np.ndarray,
                             W_in: np.ndarray, W_out: np.ndarray,
                             input_idx: np.ndarray, output_idx: np.ndarray,
                             x_seq: np.ndarray, dt: float = 1.0, tau: float = 5.0):
    """x_seq: (T, n_input) 시퀀스를 넣고 (T, n_output) 출력을 얻는다.
    h0=0에서 시작해 순차적으로 업데이트."""
    n = A_signed.shape[0]
    h = np.zeros(n, dtype=np.float32)
    W_eff = A_signed * scale  # 토폴로지(A_signed) x 학습된 세기(scale)
    outputs = []

    for x_t in x_seq:
        drive = np.zeros(n, dtype=np.float32)
        drive[input_idx] += W_in @ x_t
        rec_input = W_eff.T @ h  # h_j가 A[j,i]를 통해 뉴런 i로 들어옴 -> A.T @ h
        pre_activation = rec_input + drive
        target = np.tanh(pre_activation)
        h = h + (dt / tau) * (-h + target)
        outputs.append(W_out @ h[output_idx])

    return np.stack(outputs, axis=0)


def test_reference_numpy():
    """토폴로지·인덱스 배선이 맞는지 최소한의 스모크 테스트.

    주의: 작은 랜덤 희소그래프(50개 뉴런, 5% 밀도)라 입력->출력 사이 홉수가
    멀면 신호가 극도로 작게(1e-8 스케일) 나올 수 있다 — 이건 버그가 아니라
    '무작위로 만든 작은 테스트 그래프의 신호 감쇠' 현상이다. 그래서 여기서는
    입력/출력 뉴런 그룹을 서로 가깝게 배치해, 몇 스텝 안에 신호가 확실히
    도달하는지를 보는 것으로 테스트를 구성했다. 실제 hemibrain 데이터는
    수만 개 뉴런에 조밀한 실제 해부학적 경로가 있어 이런 문제가 없다.
    """
    rng = np.random.default_rng(0)
    n, n_in, n_out, T = 50, 5, 3, 20
    A_signed = (rng.random((n, n)) < 0.05).astype(np.float32) * rng.normal(size=(n, n)).astype(np.float32)
    # 입력 뉴런(0~4) -> 출력 뉴런(5~7) 사이에 강제로 직접 경로를 몇 개 심어서
    # "배선이 맞으면 신호가 확실히 전달된다"를 눈으로 확인할 수 있게 함
    input_idx = np.arange(0, n_in)
    output_idx = np.arange(n_in, n_in + n_out)
    for i in input_idx:
        for j in output_idx:
            A_signed[i, j] = rng.normal(2.0, 0.3)

    scale = np.abs(rng.normal(0.3, 0.05, size=(n, n))).astype(np.float32)
    W_in = rng.normal(size=(n_in, n_in)).astype(np.float32)
    W_out = rng.normal(size=(n_out, n_out)).astype(np.float32)
    x_seq = np.ones((T, n_in), dtype=np.float32)  # 일정한 입력을 넣고 신호가 자라는지 확인

    y = reference_forward_numpy(A_signed, scale, W_in, W_out, input_idx, output_idx, x_seq)
    assert y.shape == (T, n_out), f"출력 shape 오류: {y.shape}"
    assert np.all(np.isfinite(y)), "출력에 NaN/Inf 발생"
    grew = np.abs(y[-1]).sum() > np.abs(y[0]).sum()
    print(f"[connectome_rnn] numpy 참조구현 스모크테스트 통과 — 출력 shape={y.shape}")
    print(f"  t=0  출력: {y[0]}")
    print(f"  t={T-1} 출력: {y[-1]}")
    print(f"  신호가 시간에 따라 커짐(=배선을 타고 전파됨): {grew}")
    return y


# ---------------------------------------------------------------------- #
# torch 버전 (실제 학습용)
# ---------------------------------------------------------------------- #
if _HAS_TORCH:

    class ConnectomeRNN(nn.Module):
        """커넥톰 토폴로지로 고정된 재귀신경망.

        Parameters
        ----------
        A_signed : (N, N) numpy 배열 — 커넥톰에서 온 부호있는 고정 토폴로지
                   (0이 아닌 곳만 연결 존재, 학습으로도 0이 새로 생기거나 없던
                   연결이 생기지 않는다 — buffer로 등록해 고정)
        input_idx, output_idx : 입력/출력으로 쓸 뉴런 인덱스
        n_obs, n_act : 외부 관측/행동 차원 (LiDAR 차원, 조향+가속 등)
        dt, tau : 적분 스텝, 시정수
        """

        def __init__(self, A_signed: np.ndarray, input_idx: np.ndarray, output_idx: np.ndarray,
                     n_obs: int, n_act: int, dt: float = 1.0, tau: float = 5.0):
            super().__init__()
            n = A_signed.shape[0]
            self.n = n
            self.dt, self.tau = dt, tau

            mask = (A_signed != 0).astype(np.float32)
            self.register_buffer("A_signed", torch.tensor(A_signed, dtype=torch.float32))
            self.register_buffer("topo_mask", torch.tensor(mask, dtype=torch.float32))
            self.register_buffer("input_idx", torch.tensor(input_idx, dtype=torch.long))
            self.register_buffer("output_idx", torch.tensor(output_idx, dtype=torch.long))

            # 토폴로지는 고정(buffer), "세기(scale)"만 학습 가능한 파라미터.
            # topo_mask를 곱해서, 원래 연결이 없던 자리엔 학습 중에도 절대
            # 값이 생기지 않도록 강제한다 (매 forward에서 마스킹).
            self.scale = nn.Parameter(torch.ones(n, n) * 0.05)

            self.W_in = nn.Linear(n_obs, len(input_idx))
            self.W_out = nn.Linear(len(output_idx), n_act)

        def effective_weight(self):
            return self.A_signed * self.scale * self.topo_mask

        def forward(self, x: torch.Tensor, h: torch.Tensor | None = None, n_steps: int = 3):
            """x: (batch, n_obs) 한 스텝의 관측. n_steps: 내부 적분 스텝 수
            (관측 한 번에 뇌 안에서 정보가 여러 스텝 확산되도록)."""
            batch = x.shape[0]
            if h is None:
                h = torch.zeros(batch, self.n, device=x.device)

            drive = torch.zeros(batch, self.n, device=x.device)
            drive[:, self.input_idx] = self.W_in(x)

            W_eff = self.effective_weight()
            for _ in range(n_steps):
                rec_input = h @ W_eff  # (batch,n) @ (n,n) -> (batch,n), A[i,j]: i->j
                pre_act = rec_input + drive
                target = torch.tanh(pre_act)
                h = h + (self.dt / self.tau) * (-h + target)

            y = self.W_out(h[:, self.output_idx])
            return y, h


if __name__ == "__main__":
    test_reference_numpy()
