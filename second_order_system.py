import numpy as np
import matplotlib.pyplot as plt
import control

class SecondOrderSystem:
    def __init__(self, 
                zeta=None, 
                omega_n=None, 
                m=None, 
                b=None, 
                k=None,
                kp=0.0,
                ki=0.0,
                kd=0.0,
                disturbance=0.0):
        has_zw = zeta is not None and omega_n is not None
        has_mbk = m is not None and b is not None and k is not None
        

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.disturbance = disturbance

        if has_zw and not has_mbk:
            self.zeta = zeta
            self.omega_n = omega_n
            self.m = 1.0
            self.update_bk()
        elif has_mbk and not has_zw:
            self.m = m
            self.b = b
            self.k = k
            self.update_zeta_omega_n()
        elif not has_zw and not has_mbk:
            self.m = 1.0
            self.k = 1.0
            self.b = 0.0
            self.update_zeta_omega_n()
        else:
            raise ValueError(
                "Pass either (zeta, omega_n) or (m, b, k), not a mix of both."
            )

    def gen_zeta(self):
        return self.b / (2 * self.m * self.omega_n)
    def gen_omega_n(self):
        return np.sqrt(self.k / self.m)
    def gen_b(self):
        return 2 * self.zeta * self.m * self.omega_n
    def gen_k(self):
        return self.m * self.omega_n**2

    def generate_system(self):
        
        A = np.array([[0, 1], [-self.k/self.m, -self.b/self.m]])
        B = np.array([0, 1/self.m])
        C = np.array([1, 0])
        D = np.array([0])
        self.plant_sys = control.StateSpace(A, B, C, D)

        tau = 0.01
        self.controller_num = [self.kd, self.kp, self.ki]
        self.controller_den = [tau, 1, 0]
        self.controller_sys = control.TransferFunction(self.controller_num, self.controller_den)
        self.closed_loop_sys = control.feedback(self.plant_sys, self.controller_sys)

    def update_zeta_omega_n(self):
        self.zeta = self.gen_zeta()
        self.omega_n = self.gen_omega_n()
        self.generate_system()
    def update_bk(self):
        self.b = self.gen_b()
        self.k = self.gen_k()
        self.generate_system()

    def update_zeta(self, zeta):
        self.zeta = zeta
        self.update_bk()
    
    def update_omega_n(self, omega_n):
        self.omega_n = omega_n
        self.update_bk()
    
    def update_m(self, m):
        self.m = m
        self.update_zeta_omega_n()
    
    def update_b(self, b):
        self.b = b
        self.update_zeta_omega_n()
    
    def update_k(self, k):
        self.k = k
        self.update_zeta_omega_n()

    def update_kp(self, kp):
        self.kp = kp
        self.generate_system()
    
    def update_ki(self, ki):
        self.ki = ki
        self.generate_system()
    
    def update_kd(self, kd):
        self.kd = kd
        self.generate_system()
    
    def update_disturbance(self, disturbance):
        self.disturbance = disturbance
        self.generate_system()

    def plot_response(self, T=10, x0=[1, 0, 0]):
        time = np.linspace(0, T, 1000)
        dist = self.disturbance * np.ones(len(time))
        response = control.forced_response(self.closed_loop_sys, T=time, U=dist, X0=x0)
        
        plt.plot(response.time, response.states[0])
        plt.show()

if __name__ == "__main__":
    system = SecondOrderSystem(zeta=0.5, omega_n=1.0, 
                               kp=1.0, ki=0.0, kd=0.0, 
                               disturbance=-1.0)
    print(system.closed_loop_sys)
    system.plot_response()
