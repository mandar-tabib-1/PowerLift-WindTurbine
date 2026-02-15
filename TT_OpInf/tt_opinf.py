"""
TT-OpInf: Tensor Train + Operator Inference for Parametric ROM

A simplified approach that directly applies TT decomposition to the 
original 4D tensor without GCA-ROM. This is the straightforward approach:

Pipeline:
---------
Offline (Training):
    1. Normalize data
    2. Apply TT decomposition: tensor_4d → G1 ⊗ G2 ⊗ G3 (param × time × space)
    3. Train OpInf on time coefficients for temporal dynamics

Online (Inference):
    1. Interpolate G1 (parameter core) via RBF for new yaw angle
    2. Use OpInf to predict time coefficients
    3. Reconstruct full field via TT cores
    4. Denormalize

Author: Auto-generated
Date: 2024
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from scipy.interpolate import RBFInterpolator
import warnings


class TensorTrainDecomposition:
    """
    Tensor Train decomposition for 4D tensors: [params, time, space, features]
    
    Decomposes tensor as: T ≈ G1 ⊗ G2 ⊗ G3
    where:
        G1: Parameter core (N_params, r1)
        G2: Time core (r1, N_time, r2)  
        G3: Spatial core (r2, N_space * N_features)
    """
    
    def __init__(self, tolerance: float = 1e-6, max_rank: Optional[int] = None):
        """
        Args:
            tolerance: Relative tolerance for truncated SVD
            max_rank: Maximum TT rank (None = no limit)
        """
        self.tolerance = tolerance
        self.max_rank = max_rank
        self.G1 = None  # Parameter core
        self.G2 = None  # Time core
        self.G3 = None  # Spatial core
        self.ranks = None
        self.original_shape = None
        
    def fit(self, tensor_4d: np.ndarray) -> 'TensorTrainDecomposition':
        """
        Compute TT decomposition of 4D tensor.
        
        Args:
            tensor_4d: Input tensor of shape (N_params, N_time, N_space, N_features)
            
        Returns:
            self with fitted cores
        """
        if isinstance(tensor_4d, np.ndarray):
            data = tensor_4d
        else:
            data = tensor_4d.numpy()
            
        self.original_shape = data.shape
        N_params, N_time, N_space, N_features = data.shape
        
        # Reshape to 3D: combine space and features
        # (N_params, N_time, N_space * N_features)
        data_3d = data.reshape(N_params, N_time, -1)
        
        # First unfolding: parameter mode
        # (N_params, N_time * N_space * N_features)
        X1 = data_3d.reshape(N_params, -1)
        
        # SVD for first split
        U1, S1, Vt1 = np.linalg.svd(X1, full_matrices=False)
        
        # Determine rank
        r1 = self._compute_rank(S1)
        
        # G1: Parameter core (N_params, r1)
        self.G1 = U1[:, :r1]
        
        # Intermediate: (r1, N_time * N_space * N_features)
        intermediate = np.diag(S1[:r1]) @ Vt1[:r1, :]
        
        # Reshape for second split: (r1 * N_time, N_space * N_features)
        X2 = intermediate.reshape(r1, N_time, -1).transpose(0, 1, 2).reshape(r1 * N_time, -1)
        
        # SVD for second split
        U2, S2, Vt2 = np.linalg.svd(X2, full_matrices=False)
        
        # Determine rank
        r2 = self._compute_rank(S2)
        
        # G2: Time core (r1, N_time, r2)
        self.G2 = U2[:, :r2].reshape(r1, N_time, r2)
        
        # G3: Spatial core (r2, N_space * N_features)
        self.G3 = np.diag(S2[:r2]) @ Vt2[:r2, :]
        
        self.ranks = [r1, r2]
        
        print(f"TT Decomposition complete:")
        print(f"  Original shape: {self.original_shape}")
        print(f"  G1 (param): {self.G1.shape}")
        print(f"  G2 (time): {self.G2.shape}")
        print(f"  G3 (space): {self.G3.shape}")
        print(f"  Ranks: r1={r1}, r2={r2}")
        print(f"  Compression: {data.size} → {self.G1.size + self.G2.size + self.G3.size}")
        
        return self
    
    def _compute_rank(self, singular_values: np.ndarray) -> int:
        """Compute truncation rank based on tolerance."""
        total_energy = np.sum(singular_values**2)
        cumulative = np.cumsum(singular_values**2) / total_energy
        rank = np.searchsorted(cumulative, 1 - self.tolerance**2) + 1
        rank = min(rank, len(singular_values))
        if self.max_rank is not None:
            rank = min(rank, self.max_rank)
        return max(1, rank)
    
    def reconstruct(self, param_idx: int = None, time_idx: int = None, 
                   g1_custom: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct tensor from TT cores.
        
        Args:
            param_idx: Parameter index (use all if None)
            time_idx: Time index (use all if None)
            g1_custom: Custom G1 for interpolated parameter
            
        Returns:
            Reconstructed data
        """
        g1 = g1_custom if g1_custom is not None else self.G1[param_idx:param_idx+1] if param_idx is not None else self.G1
        
        # Contract G1 with G2
        # g1: (n_p, r1), G2: (r1, N_time, r2)
        temp = np.einsum('pr,rtq->ptq', g1, self.G2)  # (n_p, N_time, r2)
        
        if time_idx is not None:
            temp = temp[:, time_idx:time_idx+1, :]
        
        # Contract with G3
        # temp: (n_p, n_t, r2), G3: (r2, N_space * N_features)
        result = np.einsum('ptq,qs->pts', temp, self.G3)  # (n_p, n_t, N_space * N_features)
        
        return result.squeeze()
    
    def get_time_coefficients(self) -> np.ndarray:
        """
        Get time coefficients for dynamics training.
        
        Returns:
            g_time: (N_time, r1 * r2) - flattened time coefficients
        """
        # G2: (r1, N_time, r2)
        r1, N_time, r2 = self.G2.shape
        # Reshape to (N_time, r1 * r2)
        g_time = self.G2.transpose(1, 0, 2).reshape(N_time, -1)
        return g_time
    
    def set_time_coefficients(self, g_time: np.ndarray):
        """
        Set time coefficients (for prediction).
        
        Args:
            g_time: (N_time_new, r1 * r2)
        """
        r1, _, r2 = self.G2.shape
        N_time_new = g_time.shape[0]
        self.G2 = g_time.reshape(N_time_new, r1, r2).transpose(1, 0, 2)
    
    def interpolate_parameter(self, new_param: float, param_values: np.ndarray) -> np.ndarray:
        """
        Interpolate G1 for a new parameter value using RBF.
        
        Args:
            new_param: New parameter value
            param_values: Training parameter values
            
        Returns:
            Interpolated g1 vector of shape (r1,)
        """
        # RBF interpolation
        rbf = RBFInterpolator(param_values.reshape(-1, 1), self.G1, kernel='thin_plate_spline')
        g1_new = rbf(np.array([[new_param]])).flatten()
        return g1_new
    
    def compute_reconstruction_error(self, tensor_4d: np.ndarray) -> float:
        """Compute relative reconstruction error."""
        if isinstance(tensor_4d, np.ndarray):
            data = tensor_4d
        else:
            data = tensor_4d.numpy()
        recon = self.reconstruct().reshape(data.shape)
        error = np.linalg.norm(recon - data) / np.linalg.norm(data)
        return error


class OperatorInference:
    """
    Operator Inference (OpInf) for learning temporal dynamics.
    
    Learns polynomial dynamical system:
        dg/dt = c + A₁ @ g + A₂ @ (g ⊗ g)
    
    from time series data via regularized least-squares.
    """
    
    def __init__(self, polynomial_degree: int = 2, regularization: float = 1e-6,
                 include_constant: bool = True):
        """
        Args:
            polynomial_degree: Max polynomial order (1=linear, 2=quadratic)
            regularization: Tikhonov regularization parameter
            include_constant: Whether to include constant term
        """
        self.polynomial_degree = polynomial_degree
        self.regularization = regularization
        self.include_constant = include_constant
        self.operators = None
        self.dt = None
        self.n_features = None
        
    def _compute_derivatives(self, g: np.ndarray, dt: float) -> np.ndarray:
        """Compute time derivatives using central finite differences."""
        g_dot = (g[2:] - g[:-2]) / (2 * dt)
        return g_dot
    
    def _build_data_matrix(self, g: np.ndarray) -> np.ndarray:
        """Build data matrix for least-squares regression."""
        N_time, r = g.shape
        terms = []
        
        if self.include_constant:
            terms.append(np.ones((N_time, 1)))
        
        terms.append(g)
        
        if self.polynomial_degree >= 2:
            quad_terms = []
            for i in range(r):
                for j in range(i, r):
                    quad_terms.append(g[:, i:i+1] * g[:, j:j+1])
            if quad_terms:
                terms.append(np.hstack(quad_terms))
        
        return np.hstack(terms)
    
    def fit(self, g: np.ndarray, dt: float) -> 'OperatorInference':
        """
        Fit OpInf model to time series data.
        
        Args:
            g: Time series of shape (N_time, r)
            dt: Time step
        """
        self.dt = dt
        self.n_features = g.shape[1]
        
        # Compute derivatives (interior points)
        g_dot = self._compute_derivatives(g, dt)
        
        # Build data matrix
        D = self._build_data_matrix(g[1:-1])
        
        # Regularized least-squares
        n_terms = D.shape[1]
        reg_matrix = self.regularization * np.eye(n_terms)
        self.operators = np.linalg.solve(D.T @ D + reg_matrix, D.T @ g_dot)
        
        # Compute fit error
        g_dot_pred = D @ self.operators
        fit_error = np.linalg.norm(g_dot_pred - g_dot) / np.linalg.norm(g_dot)
        print(f"OpInf fit error: {fit_error:.6e}")
        
        return self
    
    def _form_feature_matrix(self, g: np.ndarray) -> np.ndarray:
        """Form feature matrix for prediction."""
        return self._build_data_matrix(g)
    
    def predict_derivative(self, g: np.ndarray) -> np.ndarray:
        """Predict dg/dt given current state g."""
        if g.ndim == 1:
            g = g.reshape(1, -1)
        X = self._form_feature_matrix(g)
        return (X @ self.operators).flatten()
    
    def integrate(self, g0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        """
        Integrate dynamics using RK4.
        
        Args:
            g0: Initial condition
            n_steps: Number of time steps
            dt: Time step
            
        Returns:
            Trajectory of shape (n_steps + 1, r)
        """
        trajectory = [g0.copy()]
        g = g0.copy()
        
        for _ in range(n_steps):
            k1 = self.predict_derivative(g)
            k2 = self.predict_derivative(g + 0.5 * dt * k1)
            k3 = self.predict_derivative(g + 0.5 * dt * k2)
            k4 = self.predict_derivative(g + dt * k3)
            g = g + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(g.copy())
            
        return np.array(trajectory)
    
    def save(self, filepath: str):
        """Save OpInf model."""
        np.savez(filepath,
                 operators=self.operators,
                 dt=self.dt,
                 n_features=self.n_features,
                 polynomial_degree=self.polynomial_degree,
                 regularization=self.regularization,
                 include_constant=self.include_constant)
        
    @classmethod
    def load(cls, filepath: str) -> 'OperatorInference':
        """Load OpInf model."""
        data = np.load(filepath)
        model = cls(
            polynomial_degree=int(data['polynomial_degree']),
            regularization=float(data['regularization']),
            include_constant=bool(data['include_constant'])
        )
        model.operators = data['operators']
        model.dt = float(data['dt'])
        model.n_features = int(data['n_features'])
        return model


class TT_OpInf:
    """
    Tensor Train + Operator Inference (TT-OpInf)
    
    Simple, direct approach for parametric spatio-temporal modeling:
    - TT decomposition separates parameter, time, and spatial modes
    - RBF interpolation for new parameter values
    - OpInf for temporal dynamics prediction
    
    No GCA-ROM, no neural networks - just classical numerical methods.
    """
    
    def __init__(
        self,
        tt_tolerance: float = 1e-6,
        tt_max_rank: Optional[int] = None,
        opinf_degree: int = 2,
        opinf_regularization: float = 1e-6
    ):
        """
        Initialize TT-OpInf model.
        
        Args:
            tt_tolerance: Tolerance for TT decomposition
            tt_max_rank: Maximum TT rank
            opinf_degree: Polynomial degree for OpInf (1 or 2)
            opinf_regularization: Regularization for OpInf
        """
        self.tt_tolerance = tt_tolerance
        self.tt_max_rank = tt_max_rank
        self.opinf_degree = opinf_degree
        self.opinf_regularization = opinf_regularization
        
        # Components (initialized during fit)
        self.tt_decomp = None
        self.dynamics_model = None
        
        # Data info
        self.param_values = None
        self.dt = None
        self.n_features = None
        self.n_space = None
        self.n_time_train = None
        self.data_min = None
        self.data_max = None
        
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1]."""
        return (data - self.data_min) / (self.data_max - self.data_min + 1e-10)
    
    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data from [0, 1]."""
        return data * (self.data_max - self.data_min) + self.data_min
    
    def fit(
        self,
        tensor_4d,
        param_values: np.ndarray,
        dt: float,
        normalize: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Fit TT-OpInf model.
        
        Args:
            tensor_4d: 4D tensor of shape (N_params, N_time, N_space, N_features)
            param_values: Parameter values for each slice
            dt: Time step
            normalize: Whether to normalize data
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        # Convert to numpy
        if hasattr(tensor_4d, 'numpy'):
            tensor_4d = tensor_4d.numpy()
        
        self.param_values = np.array(param_values)
        self.dt = dt
        
        N_params, N_time, N_space, N_features = tensor_4d.shape
        self.n_features = N_features
        self.n_space = N_space
        self.n_time_train = N_time
        
        history = {'tt_error': [], 'opinf_error': []}
        
        if verbose:
            print("=" * 60)
            print("TT-OpInf: Tensor Train + Operator Inference")
            print("=" * 60)
            print(f"Input shape: {tensor_4d.shape}")
            print(f"  Parameters: {N_params} ({param_values})")
            print(f"  Time steps: {N_time}")
            print(f"  Spatial nodes: {N_space}")
            print(f"  Features: {N_features}")
        
        # =================================================================
        # Stage 1: Normalize
        # =================================================================
        if normalize:
            self.data_min = tensor_4d.min()
            self.data_max = tensor_4d.max()
            tensor_4d = self._normalize(tensor_4d)
            if verbose:
                print(f"\nData normalized to [0, 1]")
        else:
            self.data_min = 0.0
            self.data_max = 1.0
        
        # =================================================================
        # Stage 2: TT Decomposition
        # =================================================================
        if verbose:
            print("\n" + "-" * 60)
            print("Stage 1: Tensor Train Decomposition")
            print("-" * 60)
        
        self.tt_decomp = TensorTrainDecomposition(
            tolerance=self.tt_tolerance,
            max_rank=self.tt_max_rank
        )
        self.tt_decomp.fit(tensor_4d)
        
        tt_error = self.tt_decomp.compute_reconstruction_error(tensor_4d)
        history['tt_error'].append(tt_error)
        
        if verbose:
            print(f"TT reconstruction error: {tt_error:.6e}")
        
        # =================================================================
        # Stage 3: Train OpInf on time coefficients
        # =================================================================
        if verbose:
            print("\n" + "-" * 60)
            print("Stage 2: Operator Inference for Dynamics")
            print("-" * 60)
        
        # Get time coefficients from TT decomposition
        g_time = self.tt_decomp.get_time_coefficients()
        
        if verbose:
            print(f"Time coefficients shape: {g_time.shape}")
        
        self.dynamics_model = OperatorInference(
            polynomial_degree=self.opinf_degree,
            regularization=self.opinf_regularization
        )
        self.dynamics_model.fit(g_time, dt)
        
        if verbose:
            print("\n" + "=" * 60)
            print("TT-OpInf Training Complete!")
            print("=" * 60)
            print(f"  TT ranks: {self.tt_decomp.ranks}")
            print(f"  OpInf dimension: {self.dynamics_model.n_features}")
        
        return history
    
    def predict(
        self,
        param_value: float,
        n_time_steps: Optional[int] = None,
        dt: Optional[float] = None,
        return_latent: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict spatio-temporal field for a new parameter value.
        
        Args:
            param_value: New parameter value (e.g., yaw angle)
            n_time_steps: Number of time steps to predict
            dt: Time step (default: training dt)
            return_latent: Whether to return time coefficients
            
        Returns:
            Predicted field of shape (N_time, N_space, N_features)
        """
        if n_time_steps is None:
            n_time_steps = self.n_time_train - 1
        if dt is None:
            dt = self.dt
            
        # Step 1: Interpolate parameter core (G1) for new yaw
        g1_new = self.tt_decomp.interpolate_parameter(param_value, self.param_values)
        
        # Step 2: Get initial time coefficient
        g_time = self.tt_decomp.get_time_coefficients()
        g0 = g_time[0]  # Initial condition
        
        # Step 3: Integrate dynamics with OpInf
        g_trajectory = self.dynamics_model.integrate(g0, n_time_steps, dt)
        
        # Step 4: Reconstruct field using TT cores
        # Save original G2 before modifying
        original_G2 = self.tt_decomp.G2.copy()
        
        # Set predicted time coefficients
        self.tt_decomp.set_time_coefficients(g_trajectory)
        
        # Create custom G1 for new parameter
        g1_custom = g1_new[np.newaxis, :]  # (1, r1)
        
        # Reconstruct
        predictions_flat = self.tt_decomp.reconstruct(g1_custom=g1_custom)
        
        # Restore original G2
        self.tt_decomp.G2 = original_G2
        
        # Reshape: (N_time, N_space, N_features)
        predictions = predictions_flat.reshape(-1, self.n_space, self.n_features)
        
        # Denormalize
        predictions = self._denormalize(predictions)
        
        if return_latent:
            return predictions, g_trajectory
        return predictions
    
    def predict_interpolation_only(
        self,
        param_value: float,
        time_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict using only parameter interpolation (no OpInf).
        Uses original time coefficients from TT decomposition.
        
        Args:
            param_value: New parameter value
            time_idx: Specific time index (None = all times)
            
        Returns:
            Predicted field
        """
        # Interpolate G1
        g1_new = self.tt_decomp.interpolate_parameter(param_value, self.param_values)
        g1_custom = g1_new[np.newaxis, :]
        
        # Reconstruct with original time core
        predictions_flat = self.tt_decomp.reconstruct(time_idx=time_idx, g1_custom=g1_custom)
        
        if time_idx is not None:
            predictions = predictions_flat.reshape(self.n_space, self.n_features)
        else:
            predictions = predictions_flat.reshape(-1, self.n_space, self.n_features)
        
        return self._denormalize(predictions)
    
    def reconstruct_training(self, param_idx: int) -> np.ndarray:
        """
        Reconstruct training data for a given parameter.
        
        Args:
            param_idx: Index in training parameters
            
        Returns:
            Reconstructed field of shape (N_time, N_space, N_features)
        """
        recon_flat = self.tt_decomp.reconstruct(param_idx=param_idx)
        recon = recon_flat.reshape(-1, self.n_space, self.n_features)
        return self._denormalize(recon)
    
    def save(self, save_dir: str):
        """Save the TT-OpInf model."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save TT decomposition
        np.savez(os.path.join(save_dir, 'tt_decomp.npz'),
                 G1=self.tt_decomp.G1,
                 G2=self.tt_decomp.G2,
                 G3=self.tt_decomp.G3,
                 ranks=self.tt_decomp.ranks,
                 original_shape=self.tt_decomp.original_shape)
        
        # Save OpInf
        self.dynamics_model.save(os.path.join(save_dir, 'opinf.npz'))
        
        # Save metadata
        np.savez(os.path.join(save_dir, 'metadata.npz'),
                 param_values=self.param_values,
                 dt=self.dt,
                 n_features=self.n_features,
                 n_space=self.n_space,
                 n_time_train=self.n_time_train,
                 data_min=self.data_min,
                 data_max=self.data_max,
                 tt_tolerance=self.tt_tolerance,
                 tt_max_rank=self.tt_max_rank if self.tt_max_rank else -1,
                 opinf_degree=self.opinf_degree,
                 opinf_regularization=self.opinf_regularization)
        
        print(f"Model saved to {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str) -> 'TT_OpInf':
        """Load a saved TT-OpInf model."""
        import os
        
        # Load metadata
        meta = np.load(os.path.join(save_dir, 'metadata.npz'))
        
        max_rank = int(meta['tt_max_rank'])
        if max_rank < 0:
            max_rank = None
            
        model = cls(
            tt_tolerance=float(meta['tt_tolerance']),
            tt_max_rank=max_rank,
            opinf_degree=int(meta['opinf_degree']),
            opinf_regularization=float(meta['opinf_regularization'])
        )
        
        model.param_values = meta['param_values']
        model.dt = float(meta['dt'])
        model.n_features = int(meta['n_features'])
        model.n_space = int(meta['n_space'])
        model.n_time_train = int(meta['n_time_train'])
        model.data_min = float(meta['data_min'])
        model.data_max = float(meta['data_max'])
        
        # Load TT decomposition
        tt_data = np.load(os.path.join(save_dir, 'tt_decomp.npz'))
        model.tt_decomp = TensorTrainDecomposition()
        model.tt_decomp.G1 = tt_data['G1']
        model.tt_decomp.G2 = tt_data['G2']
        model.tt_decomp.G3 = tt_data['G3']
        model.tt_decomp.ranks = list(tt_data['ranks'])
        model.tt_decomp.original_shape = tuple(tt_data['original_shape'])
        
        # Load OpInf
        model.dynamics_model = OperatorInference.load(os.path.join(save_dir, 'opinf.npz'))
        
        print(f"Model loaded from {save_dir}")
        return model
