import sys
import numpy as np
import rclpy
from rclpy.node import Node

from lancer_interfaces.msg import LandingPrediction
from unitree_go.msg import LowCmd, LowState

CAM_OFFSET_TORSO = np.array([0.0, 0.0, 0.0])  # Offset si besoin (en mètres)


class G1JointIndex:
    LeftHipPitch = 0; LeftHipRoll = 1; LeftHipYaw = 2; LeftKnee = 3; LeftAnklePitch = 4; LeftAnkleRoll = 5
    RightHipPitch = 6; RightHipRoll = 7; RightHipYaw = 8; RightKnee = 9; RightAnklePitch = 10; RightAnkleRoll = 11
    WaistYaw = 12; WaistRoll = 13; WaistPitch = 14
    LeftShoulderPitch = 15; LeftShoulderRoll = 16; LeftShoulderYaw = 17; LeftElbow = 18; LeftWristRoll = 19; LeftWristPitch = 20; LeftWristYaw = 21
    RightShoulderPitch = 22; RightShoulderRoll = 23; RightShoulderYaw = 24; RightElbow = 25; RightWristRoll = 26; RightWristPitch = 27; RightWristYaw = 28


NUM_ACTUATORS = 29
DEFAULT_POS = np.zeros(NUM_ACTUATORS)

# ------------------- POSITIONS PAR DÉFAUT -------------------
# Jambes
DEFAULT_POS[G1JointIndex.LeftHipPitch] = -0.1
DEFAULT_POS[G1JointIndex.LeftKnee] = 0.3
DEFAULT_POS[G1JointIndex.LeftAnklePitch] = -0.2
DEFAULT_POS[G1JointIndex.RightHipPitch] = -0.1
DEFAULT_POS[G1JointIndex.RightKnee] = 0.3
DEFAULT_POS[G1JointIndex.RightAnklePitch] = -0.2

# Bras Gauche
DEFAULT_POS[G1JointIndex.LeftShoulderPitch] = 0.3378
DEFAULT_POS[G1JointIndex.LeftShoulderRoll] = 0.2187
DEFAULT_POS[G1JointIndex.LeftShoulderYaw] = 0.0039 
DEFAULT_POS[G1JointIndex.LeftElbow] = 0.8950 
DEFAULT_POS[G1JointIndex.LeftWristRoll] = -0.0083 
DEFAULT_POS[G1JointIndex.LeftWristPitch] = 0.0087 
DEFAULT_POS[G1JointIndex.LeftWristYaw] = 0.0081  

# Bras Droit
DEFAULT_POS[G1JointIndex.RightShoulderPitch] = -0.4974
DEFAULT_POS[G1JointIndex.RightShoulderRoll] = -0.1781 
DEFAULT_POS[G1JointIndex.RightShoulderYaw] =  0.0262  
DEFAULT_POS[G1JointIndex.RightElbow] =  0.6493 
DEFAULT_POS[G1JointIndex.RightWristRoll] = 1.6567  
DEFAULT_POS[G1JointIndex.RightWristPitch] = 0.0275 
DEFAULT_POS[G1JointIndex.RightWristYaw] = -0.0263 

# ------------------- GAINS PD -------------------
KP_BODY, KD_BODY = 120.0, 10.0
KP_ARM, KD_ARM = 80.0, 6.0

ARM_JOINTS = [
    G1JointIndex.LeftShoulderRoll, G1JointIndex.LeftShoulderPitch, G1JointIndex.LeftShoulderYaw, G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll, G1JointIndex.LeftWristPitch, G1JointIndex.LeftWristYaw,
    G1JointIndex.RightShoulderRoll, G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderYaw, G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll, G1JointIndex.RightWristPitch, G1JointIndex.RightWristYaw
]

# ------------------- FONCTIONS CINÉMATIQUES -------------------
def _q2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)    ],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)    ],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk_and_jac(q, side='right'):
    """Cinématique directe + Jacobienne géométrique 4-DOF."""
    s = -1. if side == 'right' else 1.
    R, p = np.eye(3), np.zeros(3)
    p_j, z_j = [], []

    p += R @ np.array([0.0039563, s * 0.10021, 0.24778])
    R = R @ _q2R([0.990264, s * 0.139201, 0., 0.])
    p_j.append(p.copy()); z_j.append(R @ np.array([0., 1., 0.]))
    R = R @ _Ry(q[0])

    p += R @ np.array([0., s * 0.038, -0.013831])
    R = R @ _q2R([0.990268, -s * 0.139172, 0., 0.])
    p_j.append(p.copy()); z_j.append(R @ np.array([1., 0., 0.]))
    R = R @ _Rx(q[1])

    p += R @ np.array([0., s * 0.00624, -0.1032])
    p_j.append(p.copy()); z_j.append(R @ np.array([0., 0., 1.]))
    R = R @ _Rz(q[2])

    p += R @ np.array([0.015783, 0., -0.080518])
    p_j.append(p.copy()); z_j.append(R @ np.array([0., 1., 0.]))
    R = R @ _Ry(q[3])

    p_ee = p + R @ np.array([0.264, s * 0.00188791, -0.01])
    J = np.column_stack([np.cross(z_j[i], p_ee - p_j[i]) for i in range(4)])
    return p_ee, J


def solve_ik_4dof(p_target, q_init, side='right', max_iter=30, tol=0.001):
    """Calcule les angles cibles (q_target) pour atteindre p_target via pseudo-inverse."""
    q_sol = q_init.copy()
    step_size = 0.5  # Gain de convergence

    for _ in range(max_iter):
        p_actual, J = fk_and_jac(q_sol, side=side)
        error = p_target - p_actual
        
        if np.linalg.norm(error) < tol:
            break
            
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ error
        q_sol += step_size * dq

    return q_sol


# ------------------- NOEUD ROS 2 -------------------
class MoveRightArmNode(Node):
    def __init__(self):
        super().__init__('move_right_arm_node')
        self.get_logger().info("*** G1 Move Right Arm Node (IK Cartésien) Lancé ***")

        self.control_dt = 0.02        # Période du timer (50 Hz / 100 Hz selon le système)
        self.catch_duration = 1.0     # Temps pour atteindre la cible (s)
        self.hold_duration = 1.5      # Temps de maintien en position (s)
        self.return_duration = 2.0    # Temps de retour à la pose par défaut (s)

        # Calcul des pas de la machine à états
        self.catch_steps = int(self.catch_duration / self.control_dt)
        self.hold_steps = int(self.hold_duration / self.control_dt)
        self.return_steps = int(self.return_duration / self.control_dt)

        # État courant du robot
        self.low_state = None
        self.is_busy = False
        self.target_p_torso = None
        self.catch_sequence_step = 0
        self.step_counter = 0
        self.start_return_pos = None

        # Subscriptions
        self.sub_lowstate = self.create_subscription(
            LowState,
            '/lowstate',
            self.lowstate_callback,
            10
        )

        self.sub_landing = self.create_subscription(
            LandingPrediction,
            '/trajectory/landing_prediction',
            self.landing_callback,
            10
        )

        # Publisher
        self.pub_lowcmd = self.create_publisher(
            LowCmd,
            '/lowcmd',
            10
        )

        # Timer de contrôle principal
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)

    def lowstate_callback(self, msg: LowState):
        self.low_state = msg

    def landing_callback(self, msg: LandingPrediction):
        p_cam = np.array([msg.x_landing, msg.y_landing, 0.0])
        new_target = p_cam + CAM_OFFSET_TORSO

        if self.is_busy:
            self.get_logger().info(f"[UPDATE] Target updated on-the-fly: {new_target}")
        else:
            self.get_logger().info(f"[NEW CATCH] Target set: {new_target}")

        self.target_p_torso = new_target
        self.is_busy = True
        self.catch_sequence_step = 1  # Lancer la séquence
        self.step_counter = 0

    def publish_lowcmd(self, target_pos, arm_torques=None):
        cmd = LowCmd()
        
        for i in range(NUM_ACTUATORS):
            cmd.motor_cmd[i].q = float(target_pos[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            
            if i in ARM_JOINTS:
                cmd.motor_cmd[i].kp = float(KP_ARM)
                cmd.motor_cmd[i].kd = float(KD_ARM)
            else:
                cmd.motor_cmd[i].kp = float(KP_BODY)
                cmd.motor_cmd[i].kd = float(KD_BODY)
            cmd.motor_cmd[i].mode = 1

        if arm_torques is not None:
            right_arm_indices = [
                G1JointIndex.RightShoulderPitch,
                G1JointIndex.RightShoulderRoll,
                G1JointIndex.RightShoulderYaw,
                G1JointIndex.RightElbow,
            ]
            for idx, tau_val in zip(right_arm_indices, arm_torques):
                cmd.motor_cmd[idx].kp = 0.0
                cmd.motor_cmd[idx].kd = 0.0
                cmd.motor_cmd[idx].tau = float(np.clip(tau_val, -10.0, 10.0))

        self.pub_lowcmd.publish(cmd)

    def control_loop(self):
        if self.low_state is None:
            return

        # Séquence de repos / Maintien de position par défaut
        if not self.is_busy:
            self.publish_lowcmd(DEFAULT_POS)
            return

        # --- SÉQUENCE DE RATTRAPAGE PAS-À-PAS ---
        
        # 1. Déplacement vers l'objectif via Cinématique Inverse (IK)
        if self.catch_sequence_step == 1:
            self.execute_cartesian_ik_step(self.target_p_torso)
            self.step_counter += 1
            if self.step_counter >= self.catch_steps:
                self.catch_sequence_step = 2
                self.step_counter = 0
                self.get_logger().info("Target reached. Holding position...")

        # 2. Maintien de la position cible (IK)
        elif self.catch_sequence_step == 2:
            self.execute_cartesian_ik_step(self.target_p_torso)
            self.step_counter += 1
            if self.step_counter >= self.hold_steps:
                self.catch_sequence_step = 3
                self.step_counter = 0
                # Mémorisation de la position de départ pour une interpolation fluide au retour
                self.start_return_pos = np.array([self.low_state.motor_state[i].q for i in range(NUM_ACTUATORS)])
                self.get_logger().info("Returning to default pose...")

        # 3. Retour fluide vers la position par défaut
        elif self.catch_sequence_step == 3:
            alpha = self.step_counter / float(self.return_steps)
            interp_pos = (1.0 - alpha) * self.start_return_pos + alpha * DEFAULT_POS
            self.publish_lowcmd(interp_pos)
            
            self.step_counter += 1
            if self.step_counter >= self.return_steps:
                self.is_busy = False
                self.catch_sequence_step = 0
                self.get_logger().info("Back to default pose. Ready for next prediction.")

    def execute_cartesian_ik_step(self, p_target):
        """Calcule l'IK à chaque pas pour envoyer la consigne angulaire au bras droit."""
        # 1. Lecture des angles actuels des 4 articulations du bras droit
        q1 = self.low_state.motor_state[G1JointIndex.RightShoulderPitch].q
        q2 = self.low_state.motor_state[G1JointIndex.RightShoulderRoll].q
        q3 = self.low_state.motor_state[G1JointIndex.RightShoulderYaw].q
        q4 = self.low_state.motor_state[G1JointIndex.RightElbow].q
        q_current = np.array([q1, q2, q3, q4])

        # 2. Inversion cinématique
        q_target_arm = solve_ik_4dof(p_target, q_current, side='right')

        # 3. Préparation du vecteur global des positions (29 DoF)
        target_pos = DEFAULT_POS.copy()
        right_arm_indices = [
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
        ]
        for idx, q_val in zip(right_arm_indices, q_target_arm):
            target_pos[idx] = q_val

        # 4. Envoi en contrôle de POSITION (arm_torques = None)
        self.publish_lowcmd(target_pos, arm_torques=None)


def main(args=None):
    rclpy.init(args=args)
    node = MoveRightArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()