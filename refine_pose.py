import gtsam  # We can use GTSAM library for pose-graph optimization
import numpy as np

# def optimize_multicam_poses(T1_init, T2_init, T3_init, E12, E23, E31):
#     # Create a new factor graph
#     graph = gtsam.NonlinearFactorGraph()
    
#     # Create noise models
#     pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))  # for poses
#     constraint_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))  # for extrinsics
    
#     # Add prior factors to keep poses close to initial estimates
#     graph.add(gtsam.PriorFactorPose3(1, gtsam.Pose3(T1_init), pose_noise))
#     graph.add(gtsam.PriorFactorPose3(2, gtsam.Pose3(T2_init), pose_noise))
#     graph.add(gtsam.PriorFactorPose3(3, gtsam.Pose3(T3_init), pose_noise))
    
#     # Add between factors for extrinsic constraints
#     graph.add(gtsam.BetweenFactorPose3(1, 2, gtsam.Pose3(E12), constraint_noise))
#     graph.add(gtsam.BetweenFactorPose3(2, 3, gtsam.Pose3(E23), constraint_noise))
#     graph.add(gtsam.BetweenFactorPose3(3, 1, gtsam.Pose3(E31), constraint_noise))
    
#     # Create initial estimate
#     initial_estimate = gtsam.Values()
#     initial_estimate.insert(1, gtsam.Pose3(T1_init))
#     initial_estimate.insert(2, gtsam.Pose3(T2_init))
#     initial_estimate.insert(3, gtsam.Pose3(T3_init))
    
#     # Optimize
#     optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
#     result = optimizer.optimize()
    
#     # Extract results
#     T1_opt = result.atPose3(1).matrix()
#     T2_opt = result.atPose3(2).matrix()
#     T3_opt = result.atPose3(3).matrix()
    
#     return T1_opt, T2_opt, T3_opt



import gtsam
import numpy as np

# def optimize_multicam_poses(T1_init, T2_init, T3_init, E12, E23, E31):
#     """
#     Optimize the poses of three cameras given initial estimates and extrinsic constraints.

#     Parameters:
#     - T1_init, T2_init, T3_init: Initial estimates for the camera poses (4x4 numpy arrays).
#     - E12, E23, E31: Extrinsic constraints between the cameras (4x4 numpy arrays).

#     Returns:
#     - Optimized poses for T1, T2, T3 as gtsam.Pose3 objects.
#     """
#     # Convert initial transformations to GTSAM Pose3 objects
#     T1 = gtsam.Pose3(gtsam.Rot3(T1_init[:3, :3]), gtsam.Point3(T1_init[:3, 3]))
#     T2 = gtsam.Pose3(gtsam.Rot3(T2_init[:3, :3]), gtsam.Point3(T2_init[:3, 3]))
#     T3 = gtsam.Pose3(gtsam.Rot3(T3_init[:3, :3]), gtsam.Point3(T3_init[:3, 3]))

#     # Convert extrinsic constraints to GTSAM Pose3 objects
#     Z12 = gtsam.Pose3(gtsam.Rot3(E12[:3, :3]), gtsam.Point3(E12[:3, 3]))
#     Z23 = gtsam.Pose3(gtsam.Rot3(E23[:3, :3]), gtsam.Point3(E23[:3, 3]))
#     Z31 = gtsam.Pose3(gtsam.Rot3(E31[:3, :3]), gtsam.Point3(E31[:3, 3]))

#     # Create a NonlinearFactorGraph to store factors
#     graph = gtsam.NonlinearFactorGraph()

#     # Noise model for the constraints (assumed isotropic for simplicity)
#     noise_model = gtsam.noiseModel.Isotropic.Sigma(6, 3.0)  # 6 DOF, standard deviation 0.1
#     pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]))

#     # Add prior factors to the graph to keep poses close to initial estimates
#     graph.add(gtsam.PriorFactorPose3(1, T1, pose_noise))
#     graph.add(gtsam.PriorFactorPose3(2, T2, pose_noise))
#     graph.add(gtsam.PriorFactorPose3(3, T3, pose_noise))

#     # Add factors to the graph based on extrinsic constraints
#     graph.add(gtsam.BetweenFactorPose3(1, 2, Z12, noise_model))
#     graph.add(gtsam.BetweenFactorPose3(2, 3, Z23, noise_model))
#     graph.add(gtsam.BetweenFactorPose3(3, 1, Z31, noise_model))

#     # Create initial estimate for the poses
#     initial_estimate = gtsam.Values()
#     initial_estimate.insert(1, T1)
#     initial_estimate.insert(2, T2)
#     initial_estimate.insert(3, T3)

#     # Set up the optimizer
#     params = gtsam.LevenbergMarquardtParams()
#     params.setVerbosityLM('ERROR')  # Suppress detailed output
#     optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

#     # Optimize the poses
#     result = optimizer.optimize()

#     # Extract the optimized poses
#     T1_optimized = result.atPose3(1).matrix()
#     T2_optimized = result.atPose3(2).matrix()
#     T3_optimized = result.atPose3(3).matrix()

#     return T1_optimized, T2_optimized, T3_optimized

import torch
import numpy as np
from torch import optim

def matrix_to_axis_angle_translation(T):
    """Convert 4x4 transformation matrix to axis-angle and translation"""
    R = T[:3, :3]
    t = T[:3, 3]

    # convert R and t to torch tensors
    R = torch.tensor(R)
    t = torch.tensor(t)
    
    # Convert rotation matrix to axis-angle
    theta = torch.acos((torch.trace(R) - 1) / 2)
    if theta.abs() < 1e-6:  # Near zero rotation
        axis = torch.zeros(3)
    else:
        axis = 1/(2*torch.sin(theta)) * torch.tensor([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ])
    
    axisangle = axis * theta
    return axisangle, t

def axis_angle_translation_to_matrix(axisangle, t):
    """Convert axis-angle and translation to 4x4 transformation matrix"""
    T = torch.eye(4)
    
    theta = torch.norm(axisangle)
    if theta < 1e-6:  # Near zero rotation
        R = torch.eye(3)
    else:
        axis = axisangle / theta
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = torch.eye(3) + torch.sin(theta)*K + (1-torch.cos(theta))*K@K
    
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def optimize_multicam_poses(T1_init, T2_init, T3_init, E12, E23, E31):
    # Convert initial transforms to axis-angle and translation
    axisangle1, t1 = matrix_to_axis_angle_translation(T1_init)
    axisangle2, t2 = matrix_to_axis_angle_translation(T2_init)
    axisangle3, t3 = matrix_to_axis_angle_translation(T3_init)
    
    # Parameters to optimize: 6D vectors (3D axis-angle + 3D translation)
    params = torch.cat([
        torch.cat([axisangle1, t1]),
        torch.cat([axisangle2, t2]),
        torch.cat([axisangle3, t3])
    ])
    params.requires_grad = True
    
    optimizer = optim.Adam([params], lr=0.001)

    num_iterations = 1000
    λr, λt = 1.0, 1.0  # Weights for rotation and translation errors
    λr_reg, λt_reg = 1.0, 1.0  # Regularization weights
    
    for iter in range(num_iterations):
        optimizer.zero_grad()
        
        # Reconstruct transformation matrices
        axisangle1, t1 = params[0:3], params[3:6]
        axisangle2, t2 = params[6:9], params[9:12]
        axisangle3, t3 = params[12:15], params[15:18]
        
        T1 = axis_angle_translation_to_matrix(axisangle1, t1)
        T2 = axis_angle_translation_to_matrix(axisangle2, t2)
        T3 = axis_angle_translation_to_matrix(axisangle3, t3)
        
        # Extract rotations and translations
        R1, R2, R3 = T1[:3,:3], T2[:3,:3], T3[:3,:3]
        t1, t2, t3 = T1[:3,3], T2[:3,3], T3[:3,3]

        # convert all to torch tensors
        R1 = torch.tensor(R1)
        R2 = torch.tensor(R2)
        R3 = torch.tensor(R3)
        t1 = torch.tensor(t1)
        t2 = torch.tensor(t2)
        t3 = torch.tensor(t3)
        
        # Extract extrinsic rotations and translations
        R12, t12 = E12[:3,:3], E12[:3,3]
        R23, t23 = E23[:3,:3], E23[:3,3]
        R31, t31 = E31[:3,:3], E31[:3,3]

        # convert all to torch tensors
        R12 = torch.tensor(R12)
        R23 = torch.tensor(R23)
        R31 = torch.tensor(R31)
        t12 = torch.tensor(t12)
        t23 = torch.tensor(t23)
        t31 = torch.tensor(t31)
        
        # Compute rotation errors
        R_e12 = torch.norm(R12.T @ R1.T @ R2 - torch.eye(3))
        R_e23 = torch.norm(R23.T @ R2.T @ R3 - torch.eye(3))
        R_e31 = torch.norm(R31.T @ R3.T @ R1 - torch.eye(3))
        
        # Compute translation errors
        t_e12 = torch.norm(R12.T @ (R1.T @ (t2 - t1) - t12))
        t_e23 = torch.norm(R23.T @ (R2.T @ (t3 - t2) - t23))
        t_e31 = torch.norm(R31.T @ (R3.T @ (t1 - t3) - t31))
        
        # Regularization losses
        reg_R1 = torch.norm(axisangle1 - matrix_to_axis_angle_translation(T1_init)[0])
        reg_R2 = torch.norm(axisangle2 - matrix_to_axis_angle_translation(T2_init)[0])
        reg_R3 = torch.norm(axisangle3 - matrix_to_axis_angle_translation(T3_init)[0])
        
        reg_t1 = torch.norm(t1 - T1_init[:3,3])
        reg_t2 = torch.norm(t2 - T2_init[:3,3])
        reg_t3 = torch.norm(t3 - T3_init[:3,3])
        
        # Total loss with separate weights
        consistency_loss = (λr*(R_e12 + R_e23 + R_e31) + 
                          λt*(t_e12 + t_e23 + t_e31))
        
        reg_loss = (λr_reg*(reg_R1 + reg_R2 + reg_R3) + 
                   λt_reg*(reg_t1 + reg_t2 + reg_t3))
        
        loss = consistency_loss + reg_loss
        
        # Print losses for debugging
        if iter % 100 == 0:
            print(f"Iter {iter}, Rot Error: {R_e12 + R_e23 + R_e31:.4f}, "
                  f"Trans Error: {t_e12 + t_e23 + t_e31:.4f}, "
                  f"Total Loss: {loss.item():.4f}")
            
        loss.backward()
        optimizer.step()
        
    # Return final optimized transforms
    T1_opt = axis_angle_translation_to_matrix(params[0:3], params[3:6])
    T2_opt = axis_angle_translation_to_matrix(params[6:9], params[9:12])
    T3_opt = axis_angle_translation_to_matrix(params[12:15], params[15:18])
    
    return T1_opt, T2_opt, T3_opt




# Example usage
if __name__ == "__main__":
    # Initial pose estimates (4x4 numpy arrays)
    T_left_init = np.array([[0.9681573100463041, 0.0077328016186124, -0.2502231539706741, 0.1385640000000000],
                        [0.2462817362143865, 0.1498804162984749, 0.9575391204631920, 0.9776130000000000],
                        [0.0449080105452445, -0.9886738919326854, 0.1432033728637184, -0.5980390000000000],
                        [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    
    T_rear_init = np.array([[-0.2711431093685741, -0.9625277386806474, -0.0046654594908704, -0.1096430000000000],
                        [0.1971462909462822, -0.0602788861200971, 0.9785191852254325, 1.0225200000000001],
                        [-0.9421330873120132, 0.2643989564246421, 0.2061029782256792, -0.4819690000000000],
                        [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    
    T_right_init = np.array([[-0.8258395683402462, 0.5062981920615656, 0.2482964922804700, -0.0822707000000000],
                        [0.1468315157912473, -0.2320599695487395, 0.9615553424028607, 0.9416390000000000],
                        [0.5444534078633727, 0.8305481992324275, 0.1173037740984141, -0.4224790000000000],
                        [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

    
    def quaternion_to_matrix(qw, qx, qy, qz, tx, ty, tz):
        """Convert a quaternion and translation to a 4x4 transformation matrix."""
        T = np.eye(4)
    
        # Convert quaternion to rotation matrix
        # First row
        T[0,0] = 1 - 2*qy*qy - 2*qz*qz
        T[0,1] = 2*qx*qy - 2*qz*qw
        T[0,2] = 2*qx*qz + 2*qy*qw
        
        # Second row
        T[1,0] = 2*qx*qy + 2*qz*qw
        T[1,1] = 1 - 2*qx*qx - 2*qz*qz
        T[1,2] = 2*qy*qz - 2*qx*qw
        
        # Third row
        T[2,0] = 2*qx*qz - 2*qy*qw
        T[2,1] = 2*qy*qz + 2*qx*qw
        T[2,2] = 1 - 2*qx*qx - 2*qy*qy
        
        # Translation
        T[0,3] = tx
        T[1,3] = ty
        T[2,3] = tz
        
        return T

    # Transformation matrices for each camera
    T_car_right = quaternion_to_matrix(0.7887011425269028, -0.041220294128637876, -0.5945548165706376, -0.15085080451016103, -1.6809997757789577, 0.32260047161891275, -0.2586994713588552)
    T_car_rear = quaternion_to_matrix(-0.007335715829222361, 0.006731233479239513, -0.9921657167454186, -0.1245313947732503, 0.08940020772527049, 0.36749986187126954, -2.0581996855770064)
    T_car_left = quaternion_to_matrix(-0.7992207361115533, 0.03517299015487168, -0.5806527641924993, -0.15116693808000128, 1.63750048420827, 0.28029927779846864, -0.09049970590748779)

    # Extrinsic constraints between the cameras (4x4 numpy arrays)
    E_right_rear = np.dot(np.linalg.inv(T_car_right), np.linalg.inv(T_car_rear))
    E_rear_left = np.dot(np.linalg.inv(T_car_rear), np.linalg.inv(T_car_left))
    E_left_right = np.dot(np.linalg.inv(T_car_left), np.linalg.inv(T_car_right))

    # Optimize the poses
    T1_opt, T2_opt, T3_opt = optimize_multicam_poses(T_left_init, T_rear_init, T_right_init, E_left_right, E_rear_left, E_right_rear)

    # Print optimized poses
    print("Optimized Pose T1:")
    print(T1_opt)
    print("\nOptimized Pose T2:")
    print(T2_opt)
    print("\nOptimized Pose T3:")
    print(T3_opt)


    # compute the final extrinsics
    E12_opt = np.dot(T2_opt, np.linalg.inv(T1_opt))
    E23_opt = np.dot(T3_opt, np.linalg.inv(T2_opt))
    E31_opt = np.dot(T1_opt, np.linalg.inv(T3_opt))

    # print the extrinsics
    print("Extrinsic E_right_rear:")
    print(E_right_rear)
    print("\nExtrinsic E_rear_left:")
    print(E_rear_left)
    print("\nExtrinsic E_left_right:")
    print(E_left_right)
    
    # Print optimized extrinsics
    print("\nOptimized Extrinsic E12:")
    print(E12_opt)
    print("\nOptimized Extrinsic E23:")
    print(E23_opt)
    print("\nOptimized Extrinsic E31:")
    print(E31_opt)
