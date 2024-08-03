import mujoco
import numpy as np

class OpsaceController:

    def __init__(self, model, joint_names, eef_site_name):

        self.eef_site_name = eef_site_name
        self.joint_names = joint_names 

        self.integration_dt: float = 1.0
        self.impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
        self.impedance_ori = np.asarray([50.0, 50.0, 50.0]) # [Nm/rad]
        self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
        self.damping_ratio = 1.0
        self.Kpos: float = 0.95
        self.Kori: float = 0.95
        self.gravity_compensation: bool = True

        self.damping_pos = self.damping_ratio * 2 * np.sqrt(self.impedance_pos)
        self.damping_ori = self.damping_ratio * 2 * np.sqrt(self.impedance_ori)
        self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        self.Kd = np.concatenate([self.damping_pos, self.damping_ori], axis=0)
        self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)
        self.site_id = model.site(self.eef_site_name).id
        
        self.jac_full = np.zeros((6, model.nv))
        self.jac = np.zeros((6, 7))
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.M_inv_full = np.zeros((model.nv, model.nv))
        self.M_inv = np.zeros((7,7))
        self.Mx = np.zeros((6, 6))
        self.target_quat = np.zeros(4)

        self.actuator_ids = np.array([model.actuator(name).id for name in self.joint_names])
        self.dof_ids = np.array([model.joint(name).id for name in self.joint_names])
        self.q0 = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0])


    def get_tau(self, model, data, target_pos, target_rot, debug=False):
            
        # Spatial velocity (aka twist).
        dx = target_pos - data.site(self.site_id).xpos
        self.twist[:3] = self.Kpos * dx / self.integration_dt
        mujoco.mju_mat2Quat(self.site_quat, data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mat2Quat(self.target_quat, target_rot.flatten())
        mujoco.mju_mulQuat(self.error_quat, self.target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.Kori / self.integration_dt

        if debug:
            print("Error: ", self.twist)

        # Jacobian.
        mujoco.mj_jacSite(model, data, self.jac_full[:3], self.jac_full[3:], self.site_id)
        self.jac = self.jac_full[:,self.actuator_ids]

        # Compute the task-space inertia matrix.
        mujoco.mj_solveM(model, data, self.M_inv_full, np.eye(model.nv))
        self.M_inv = self.M_inv_full[self.actuator_ids,:]
        self.M_inv = self.M_inv[:,self.actuator_ids]
        self.Mx_inv = self.jac @ self.M_inv @ self.jac.T

        if abs(np.linalg.det(self.Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(self.Mx_inv)
        else:
            Mx = np.linalg.pinv(self.Mx_inv, rcond=1e-2)

        # Compute generalized forces.
        tau = self.jac.T @ Mx @ (self.Kp * self.twist - self.Kd * (self.jac @ data.qvel[self.dof_ids]))

        # Add joint task in nullspace.
        Jbar = self.M_inv @ self.jac.T @ Mx
        ddq = self.Kp_null * (self.q0 - data.qpos[self.dof_ids]) - self.Kd_null * data.qvel[self.dof_ids]
        tau += (np.eye(7) - self.jac.T @ Jbar.T) @ ddq

        # Add gravity compensation.
        if self.gravity_compensation:
            tau += data.qfrc_bias[self.dof_ids]

        # Set the control signal and step the simulation.
        np.clip(tau, *model.actuator_ctrlrange[:7].T, out=tau)

        return tau[self.actuator_ids]