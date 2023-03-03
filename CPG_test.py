import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class CPG:
    def __init__(self, n, R, omega, mu, a, dp, init_phi, offset):
        self.n = n          # number of channels
        self.mu = mu        # (n, 1), phase convergence rate
        self.a = a          # (n, 1), amplitude convergence rate
        self.dp = dp/2        # (n-1, 1), phase shift between neighbour joints
        self.R = R          # (n, 1), target amplitudes
        self.omega = omega  # (n, 1), target frequencies

        self.A = np.zeros((n, n))
        for i in range(n):
            self.A[i, i] = -2*self.mu[i, 0]
        self.A[0, 0] += self.mu[0, 0]
        self.A[n-1, n-1] += self.mu[n-1, 0]
        for i in range(n-1):
            self.A[i, i+1] = self.mu[i, 0]
            self.A[i+1, i] = self.mu[i + 1, 0]
        self.B = np.zeros((n, n-1))
        self.B[:n-1,:n-1] += np.eye(n-1)
        self.B[1:n, :n - 1] -= np.eye(n-1)
        self.offset = offset

        # state variables
        self.d_r = np.zeros((self.n, 1))
        self.r = self.R  # amplitude variable
        self.d_phi = np.zeros((self.n, 1))
        self.phi = init_phi     # phases

    def set_amp(self, R):
        self.R = R

    def set_freq(self, omega):
        self.omega = omega

    def set_phase_shift(self, dp):
        self.dp = dp / 2

    def update_r(self):
        # update amplitude
        dd_r = self.a*(self.a*(self.R-self.r) - self.d_r)
        self.d_r += dd_r
        self.r += self.d_r

    def update_phi(self):
        # update phase
        self.d_phi = self.omega + self.A.dot(self.phi) + self.B.dot(self.dp)
        self.phi += self.d_phi

    def update(self):
        self.update_phi()
        self.update_r()

    def reset(self, init_phi):
        self.d_r = np.zeros((self.n, 1))
        self.phi = init_phi  # phase shift
        self.r = self.R  # amplitude variable

    def output(self):
        return self.r*np.sin(self.phi)+self.offset

    def output_phase(self):
        return self.phi

if __name__ == "__main__":
    n_joints = 5
    cpg = CPG(n=n_joints,
              R=np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]),
              omega=np.array([[0.1], [0.1], [0.1], [0.1], [0.1]]),
              mu=np.array([[0.5], [0.5], [0.5], [0.5], [0.5]]),
              a=np.array([[0.1], [0.1], [0.1], [0.1], [0.1]]),
              dp=np.array([[math.pi/5], [math.pi/5], [math.pi/5], [math.pi/5]]),
              init_phi=np.zeros((n_joints, 1)),
              offset=np.zeros((n_joints, 1)))

    trail = 3   # choose one example

    if trail == 0:      # deferent frequencies
        x = np.zeros((n_joints, 1))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        cpg.set_freq(0.2*np.ones((n_joints, 1)))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        cpg.set_freq(0.05 * np.ones((n_joints, 1)))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        fig = plt.figure()
        gs = GridSpec(n_joints, 1, figure=fig)
        for i in range(n_joints):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(x[i, 1:])
        plt.title('frequencies, 0.1, 0.2, 0.05')
        plt.show()

    elif trail == 1:        # amplitude transition
        x = np.zeros((n_joints, 1))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        cpg.set_amp(2 * np.ones((n_joints, 1)))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        cpg.set_amp(3 * np.ones((n_joints, 1)))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        fig = plt.figure()
        gs = GridSpec(n_joints, 1, figure=fig)
        for i in range(n_joints):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(x[i, 1:])
        plt.title('amplitude, 1, 2, 3')
        plt.show()

    elif trail == 2: # combination of frequency and amplitude
        x = np.zeros((n_joints, 1))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        cpg.set_freq(0.2 * np.ones((n_joints, 1)))
        cpg.set_amp(2 * np.ones((n_joints, 1)))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        cpg.set_freq(0.05 * np.ones((n_joints, 1)))
        cpg.set_amp(3 * np.ones((n_joints, 1)))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        fig = plt.figure()
        gs = GridSpec(n_joints, 1, figure=fig)
        for i in range(n_joints):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(x[i, 1:])
        plt.title('frequencies, 0.1, 0.2, 0.05, amplitude, 1, 2, 3')
        plt.show()

    elif trail == 3:
        x = np.zeros((n_joints, 1))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        cpg.set_phase_shift(math.pi/2*np.ones((n_joints-1, 1)))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        cpg.set_phase_shift(math.pi * np.ones((n_joints - 1, 1)))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        fig = plt.figure()
        gs = GridSpec(n_joints, 1, figure=fig)
        for i in range(n_joints):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(x[i, 1:])
        plt.title('frequencies, 0.1, 0.2, 0.05, amplitude, 1, 2, 3')
        plt.show()

    elif trail == 4:
        # cpg.reset(np.ones((n_joints, 1)))
        x = np.zeros((n_joints, 1))
        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        for i in range(200):
            cpg.update()
            new_x = cpg.output()
            x = np.hstack((x, new_x))

        fig = plt.figure()
        gs = GridSpec(n_joints, 1, figure=fig)
        for i in range(n_joints):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(x[i, 1:])
        plt.title('initial phase')
        plt.show()

