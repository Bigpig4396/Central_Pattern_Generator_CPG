import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class CPG:
    def __init__(self, n, R, omega, mu, a, dp, init_phi, offset, dt, lower_lim=None, upper_lim=None):
        self.n = n  # number of channels
        self.dt = dt    # simulation step
        self.mu = mu  # (n, 1), phase convergence rate
        self.a = a  # (n, 1), amplitude convergence rate
        self.dp = dp  # (n-1, 1), phase shift between neighbour joints
        self.R = R  # (n, 1), target amplitudes
        self.omega = omega  # (n, 1), target frequencies

        self.A = np.zeros((n, n))
        for i in range(n):
            self.A[i, i] = -2 * self.mu[i, 0]
        self.A[0, 0] += self.mu[0, 0]
        self.A[n - 1, n - 1] += self.mu[n - 1, 0]
        for i in range(n - 1):
            self.A[i, i + 1] = self.mu[i, 0]
            self.A[i + 1, i] = self.mu[i + 1, 0]
        self.B = np.zeros((n, n - 1))
        self.B[:n - 1, :n - 1] += np.eye(n - 1)
        self.B[1:n, :n - 1] -= np.eye(n - 1)
        for i in range(n):
            self.B[i, :] *= self.mu[i, 0]
        self.offset = offset

        self.is_lower_lim = False
        if lower_lim is not None:
            self.is_lower_lim = True
            self.lower_lim = lower_lim

        self.is_upper_lim = False
        if upper_lim is not None:
            self.is_upper_lim = True
            self.upper_lim = upper_lim

        # state variables
        self.d_r = np.zeros((self.n, 1))
        # self.r = self.R  # amplitude variable, (n, 1)
        self.r = np.zeros_like(self.R)
        self.d_phi = np.zeros((self.n, 1))
        self.phi = init_phi  # phases

    def set_amp(self, R):
        self.R = R

    def set_freq(self, omega):
        self.omega = omega

    def set_phase_shift(self, dp):
        self.dp = dp

    def set_off_set(self, offset):
        self.offset = offset

    def update_r(self):
        # update amplitude
        dd_r = self.a*(self.a*(self.R-self.r) - self.d_r)
        self.d_r += dd_r * self.dt
        self.r += self.d_r * self.dt

    def update_phi(self):
        # update phase
        self.d_phi = self.omega + self.A.dot(self.phi) + self.B.dot(self.dp)
        self.phi += self.d_phi * dt

    def update(self):
        self.update_phi()
        self.update_r()

    def reset(self, init_phi):
        self.d_r = np.zeros((self.n, 1))
        self.phi = init_phi  # phase shift
        self.r = self.R  # amplitude variable

    def output(self):
        x = self.r*np.sin(self.phi)+self.offset
        if self.is_lower_lim:
            x = np.clip(x, a_min=self.lower_lim, a_max=None)
        if self.is_upper_lim:
            x = np.clip(x, a_min=None, a_max=self.upper_lim)
        return x

    def output_phase(self):
        return self.phi

    def get_dp(self):
        dp = np.zeros((self.n-1, 1))
        for i in range(self.n-1):
            dp[i, 0] = self.phi[i, 0] - self.phi[i+1, 0]
        return dp

if __name__ == "__main__":
    n_joints = 11
    dt = 0.005

    t = np.arange(0, 5, dt)
    tar_x = np.array([[60, 14, 60, 14, 60, 14, 60, 14, 60, 14, 60]]).T * math.pi / 180 * np.sin(
        np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]).T * math.pi * 2 * t
        + np.array([[0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1]]).T / 2 * math.pi)

    R = np.array([[60, 14, 60, 14, 60, 14, 60, 14, 60, 14, 60]]).T * math.pi / 180
    omega = 2 * math.pi * np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]).T
    init_phi = np.array([[0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1]]).T / 2 * math.pi
    cpg = CPG(n=n_joints,
              R=R,
              omega=omega,
              mu=5*np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).T,
              a=5*np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).T,
              dp=np.array([[0, -1, 0, -1, 0, -1, 0, -1, 0, -1]]).T / 2 * math.pi,
              init_phi=np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]).T / 2 * math.pi,
              offset=np.zeros((n_joints, 1)),
              dt=dt,
              lower_lim=-5 * np.ones((n_joints, 1)),
              upper_lim=5 * np.ones((n_joints, 1)))
    print('A', cpg.A)
    print('B', cpg.B)
    x = np.zeros((n_joints, 1))
    real_dp = np.zeros((n_joints-1, 1))
    real_r = np.zeros((n_joints, 1))
    for i in range(len(t)):
        cpg.update()
        new_x = cpg.output()
        x = np.hstack((x, new_x))
        new_dp = cpg.get_dp()
        real_dp = np.hstack((real_dp, new_dp))
        real_r = np.hstack((real_r, cpg.r))
    x = x[:, 1:]
    real_dp = real_dp[:, 1:]
    real_r = real_r[:, 1:]
    print('cpg.r', cpg.r)

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, tar_x.T, linewidth=1.0)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, x.T, linewidth=1.0)
    plt.title('comparison between the reference signal and tracking signal')
    plt.show()

    fig = plt.figure()
    gs = GridSpec(6, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, tar_x.T, linewidth=1.0)
    for i in range(5):
        ax1 = fig.add_subplot(gs[i + 1, 0])
        ax1.plot(t, tar_x.T[:, i], linewidth=1.0)
        ax1.plot(t, x.T[:, i], linewidth=1.0)
        ax1.set_title('joint ' + str(i))
    for i in range(6):
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.plot(t, tar_x.T[:, i + 5], linewidth=1.0)
        ax1.plot(t, x.T[:, i+5], linewidth=1.0)
        ax1.set_title('joint ' + str(i + 5))
    plt.title('reference')
    plt.show()

    fig = plt.figure()
    gs = GridSpec(5, 2, figure=fig)
    ref_dp = np.array([[0, -1, 0, -1, 0, -1, 0, -1, 0, -1]])
    ref_dp = np.tile(ref_dp, (x.shape[1], 1))
    for i in range(5):
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(t, real_dp.T[:, i] * 2 / math.pi, linewidth=1.0)
        ax1.plot(t, ref_dp[:, i], linewidth=1.0)
        ax1.set_title('theta ' + str(i))
        ax1.set_ylim(-2, 2)
    for i in range(5):
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.plot(t, real_dp.T[:, i + 5] * 2 / math.pi, linewidth=1.0)
        ax1.plot(t, ref_dp[:, i + 5], linewidth=1.0)
        ax1.set_title('theta ' + str(i+5))
        ax1.set_ylim(-2, 2)
    plt.title('tracking the difference of phases')
    plt.show()

    fig = plt.figure()
    gs = GridSpec(6, 2, figure=fig)
    ref_R = np.array([[60, 14, 60, 14, 60, 14, 60, 14, 60, 14, 60]])
    ref_R = np.tile(ref_R, (x.shape[1], 1))
    for i in range(6):
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(t, real_r.T[:, i], linewidth=1.0)
        ax1.plot(t, ref_R[:, i] * math.pi / 180, linewidth=1.0)
        ax1.set_title('r ' + str(i))
    for i in range(5):
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.plot(t, real_r.T[:, i + 6], linewidth=1.0)
        ax1.plot(t, ref_R[:, i + 6] * math.pi / 180, linewidth=1.0)
        ax1.set_title('r ' + str(i + 6))
    plt.title('tracking amplitude')
    plt.show()

