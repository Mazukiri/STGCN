import numpy as np

class Graph:
    """
    Định nghĩa Ma trận liền kề (Adjacency Matrix) cho 75 điểm khớp của MediaPipe.
    - 33 điểm Pose (0 -> 32)
    - 21 điểm Tay trái (33 -> 53)
    - 21 điểm Tay phải (54 -> 74)
    """
    def __init__(self):
        self.num_nodes = 75
        self.edges = self.get_edges()
        self.A = self.get_adjacency_matrix()

    def get_edges(self):
        edges = []
        
        # Liên kết giải phẫu học cơ thể (Pose) - Lấy các khớp chính của chi trên
        pose_edges = [
            (11, 12), # Vai trái nối Vai phải
            (11, 13), (13, 15), # Cánh tay trái: Vai -> Khuỷu tay -> Cổ tay
            (12, 14), (14, 16), # Cánh tay phải: Vai -> Khuỷu tay -> Cổ tay
            (11, 23), (12, 24), (23, 24) # Thân người: Vai -> Hông
        ]
        edges.extend(pose_edges)
        
        # Nối Pose với Hand
        # Cổ tay trái của Pose (15) nối với Cổ tay trái của Hand (33)
        edges.append((15, 33))
        # Cổ tay phải của Pose (16) nối với Cổ tay phải của Hand (54)
        edges.append((16, 54))
        
        # Tay trái (Left Hand: 33 -> 53)
        # Các ngón tay đều nối về cổ tay (33) hoặc nối liên tiếp nhau
        lh_edges = [
            (33, 34), (34, 35), (35, 36), (36, 37), # Thumb
            (33, 38), (38, 39), (39, 40), (40, 41), # Index
            (33, 42), (42, 43), (43, 44), (44, 45), # Middle
            (33, 46), (46, 47), (47, 48), (48, 49), # Ring
            (33, 50), (50, 51), (51, 52), (52, 53)  # Pinky
        ]
        edges.extend(lh_edges)
        
        # Tay phải (Right Hand: 54 -> 74)
        rh_edges = [
            (54, 55), (55, 56), (56, 57), (57, 58), # Thumb
            (54, 59), (59, 60), (60, 61), (61, 62), # Index
            (54, 63), (63, 64), (64, 65), (65, 66), # Middle
            (54, 67), (67, 68), (68, 69), (69, 70), # Ring
            (54, 71), (71, 72), (72, 73), (73, 74)  # Pinky
        ]
        edges.extend(rh_edges)
        
        return edges

    def get_adjacency_matrix(self):
        A = np.zeros((self.num_nodes, self.num_nodes))
        
        # Các đỉnh nối với chính nó (Self-loops)
        for i in range(self.num_nodes):
            A[i, i] = 1.0
            
        # Nối các cạnh (Vô hướng)
        for i, j in self.edges:
            A[i, j] = 1.0
            A[j, i] = 1.0
            
        # Chuẩn hóa ma trận kề (Normalized Adjacency Matrix)
        D = np.sum(A, axis=1)
        D[D == 0] = 1.0 # Tránh lỗi chia cho 0
        D_inv_sqrt = np.diag(np.power(D, -0.5))
        A_norm = np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)
        
        return A_norm
