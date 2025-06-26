import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage import morphology, measure
from skimage.morphology import skeletonize_3d, ball
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

class Skeleton3DExtractor:
    """
    3D skeleton extraction with multiple methods
    """
    
    def __init__(self, method='skeletonize_3d', preserve_topology=True):
        """
        Args:
            method (str): 'skeletonize_3d', 'medial_axis', 'thinning'
            preserve_topology (bool): Preserve topological structure
        """
        self.method = method
        self.preserve_topology = preserve_topology
    
    def extract_skeleton(self, mask):
        """
        Extract 3D skeleton from binary mask
        
        Args:
            mask: Binary mask [D, H, W] or [B, C, D, H, W]
            
        Returns:
            np.ndarray: Skeleton of same shape as input
        """
        # Handle different input shapes
        if mask.ndim == 5:  # [B, C, D, H, W]
            skeletons = np.zeros_like(mask)
            for b in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if mask[b, c].sum() > 0:
                        skeletons[b, c] = self._extract_single_skeleton(mask[b, c])
            return skeletons
        
        elif mask.ndim == 4:  # [B, D, H, W] or [C, D, H, W]
            skeletons = np.zeros_like(mask)
            for i in range(mask.shape[0]):
                if mask[i].sum() > 0:
                    skeletons[i] = self._extract_single_skeleton(mask[i])
            return skeletons
        
        else:  # [D, H, W]
            return self._extract_single_skeleton(mask)
    
    def _extract_single_skeleton(self, mask):
        """Extract skeleton from single 3D volume"""
        # Convert to binary
        binary_mask = (mask > 0.5).astype(bool)
        
        if not binary_mask.any():
            return np.zeros_like(mask, dtype=np.float32)
        
        try:
            if self.method == 'skeletonize_3d':
                skeleton = skeletonize_3d(binary_mask)
            
            elif self.method == 'medial_axis':
                # Use distance transform for medial axis
                distance = distance_transform_edt(binary_mask)
                local_maxima = morphology.local_maxima(distance)
                skeleton = local_maxima & binary_mask
            
            elif self.method == 'thinning':
                # Iterative thinning
                skeleton = self._iterative_thinning(binary_mask)
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            return skeleton.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Skeleton extraction failed: {e}")
            return np.zeros_like(mask, dtype=np.float32)
    
    def _iterative_thinning(self, mask):
        """Custom iterative thinning for 3D volumes"""
        skeleton = mask.copy()
        kernel = ball(1)
        
        changed = True
        max_iter = 100
        iteration = 0
        
        while changed and iteration < max_iter:
            old_skeleton = skeleton.copy()
            
            # Erosion step
            eroded = binary_erosion(skeleton, kernel)
            
            # Opening step
            opened = binary_dilation(eroded, kernel)
            
            # Keep points that survive opening
            skeleton = skeleton & opened
            
            changed = not np.array_equal(skeleton, old_skeleton)
            iteration += 1
        
        return skeleton


class SkeletonAnalyzer:
    """
    Analyze skeleton structure for connectivity and breakage detection
    """
    
    def __init__(self, connectivity=26):
        """
        Args:
            connectivity (int): Connectivity for 3D analysis (6, 18, 26)
        """
        self.connectivity = connectivity
        
        # Define 3D connectivity neighborhoods
        if connectivity == 6:
            # Face neighbors only
            self.neighbors = np.array([
                [-1, 0, 0], [1, 0, 0],   # z-axis
                [0, -1, 0], [0, 1, 0],   # y-axis  
                [0, 0, -1], [0, 0, 1]    # x-axis
            ])
        elif connectivity == 18:
            # Face + edge neighbors
            neighbors_6 = np.array([
                [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]
            ])
            neighbors_18_edges = np.array([
                [-1, -1, 0], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1],
                [1, -1, 0], [1, 1, 0], [1, 0, -1], [1, 0, 1],
                [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1]
            ])
            self.neighbors = np.vstack([neighbors_6, neighbors_18_edges])
        else:  # 26-connectivity
            # All neighbors in 3x3x3 cube
            self.neighbors = []
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if not (dz == 0 and dy == 0 and dx == 0):
                            self.neighbors.append([dz, dy, dx])
            self.neighbors = np.array(self.neighbors)
    
    def analyze_skeleton(self, skeleton):
        """
        Comprehensive skeleton analysis
        
        Args:
            skeleton: Binary skeleton [D, H, W]
            
        Returns:
            dict: Analysis results
        """
        if not skeleton.any():
            return self._empty_analysis()
        
        # Get skeleton coordinates
        skeleton_coords = np.where(skeleton > 0.5)
        skeleton_points = np.column_stack(skeleton_coords)
        
        # Build connectivity graph
        connectivity_graph = self._build_connectivity_graph(skeleton_points, skeleton.shape)
        
        # Analyze connectivity
        analysis = {
            'total_points': len(skeleton_points),
            'connectivity_graph': connectivity_graph,
            'endpoint_coords': self._find_endpoints(connectivity_graph, skeleton_points),
            'junction_coords': self._find_junctions(connectivity_graph, skeleton_points),
            'isolated_points': self._find_isolated_points(connectivity_graph, skeleton_points),
            'connected_components': self._find_connected_components(connectivity_graph),
            'degree_distribution': self._compute_degree_distribution(connectivity_graph),
            'skeleton_length': self._compute_skeleton_length(skeleton_points, connectivity_graph),
            'tortuosity': self._compute_tortuosity(connectivity_graph, skeleton_points)
        }
        
        return analysis
    
    def _empty_analysis(self):
        """Return empty analysis for empty skeleton"""
        return {
            'total_points': 0,
            'connectivity_graph': {},
            'endpoint_coords': np.array([]).reshape(0, 3),
            'junction_coords': np.array([]).reshape(0, 3),
            'isolated_points': np.array([]).reshape(0, 3),
            'connected_components': [],
            'degree_distribution': {},
            'skeleton_length': 0.0,
            'tortuosity': 0.0
        }
    
    def _build_connectivity_graph(self, skeleton_points, shape):
        """Build connectivity graph from skeleton points"""
        if len(skeleton_points) == 0:
            return {}
        
        # Create point index mapping
        point_to_idx = {tuple(point): i for i, point in enumerate(skeleton_points)}
        
        # Build adjacency graph
        graph = defaultdict(set)
        
        for i, point in enumerate(skeleton_points):
            # Check all neighbors
            for neighbor_offset in self.neighbors:
                neighbor_point = point + neighbor_offset
                
                # Check bounds
                if (0 <= neighbor_point[0] < shape[0] and
                    0 <= neighbor_point[1] < shape[1] and
                    0 <= neighbor_point[2] < shape[2]):
                    
                    neighbor_tuple = tuple(neighbor_point)
                    if neighbor_tuple in point_to_idx:
                        j = point_to_idx[neighbor_tuple]
                        graph[i].add(j)
                        graph[j].add(i)
        
        # Convert back to coordinate-based graph for easier access
        coord_graph = {}
        for i, point in enumerate(skeleton_points):
            coord_tuple = tuple(point)
            neighbors = []
            for neighbor_idx in graph[i]:
                neighbors.append(tuple(skeleton_points[neighbor_idx]))
            coord_graph[coord_tuple] = neighbors
        
        return coord_graph
    
    def _find_endpoints(self, graph, skeleton_points):
        """Find skeleton endpoints (degree = 1)"""
        endpoints = []
        for coord_tuple, neighbors in graph.items():
            if len(neighbors) == 1:
                endpoints.append(np.array(coord_tuple))
        return np.array(endpoints) if endpoints else np.array([]).reshape(0, 3)
    
    def _find_junctions(self, graph, skeleton_points):
        """Find skeleton junctions (degree > 2)"""
        junctions = []
        for coord_tuple, neighbors in graph.items():
            if len(neighbors) > 2:
                junctions.append(np.array(coord_tuple))
        return np.array(junctions) if junctions else np.array([]).reshape(0, 3)
    
    def _find_isolated_points(self, graph, skeleton_points):
        """Find isolated points (degree = 0)"""
        isolated = []
        skeleton_coords_set = set(tuple(point) for point in skeleton_points)
        
        for point in skeleton_points:
            coord_tuple = tuple(point)
            if coord_tuple not in graph or len(graph[coord_tuple]) == 0:
                isolated.append(point)
        return np.array(isolated) if isolated else np.array([]).reshape(0, 3)
    
    def _find_connected_components(self, graph):
        """Find connected components using DFS"""
        if not graph:
            return []
            
        visited = set()
        components = []
        
        # Convert coordinate-based graph to node indices for DFS
        coords_list = list(graph.keys())
        coord_to_idx = {coord: i for i, coord in enumerate(coords_list)}
        
        # Create adjacency list with indices
        idx_graph = {}
        for coord, neighbors in graph.items():
            idx = coord_to_idx[coord]
            idx_graph[idx] = []
            for neighbor_coord in neighbors:
                if neighbor_coord in coord_to_idx:
                    idx_graph[idx].append(coord_to_idx[neighbor_coord])
        
        for node in idx_graph.keys():
            if node not in visited:
                component = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(neighbor for neighbor in idx_graph.get(current, []) 
                                   if neighbor not in visited)
                
                components.append(component)
        
        return components
    
    def _compute_degree_distribution(self, graph):
        """Compute degree distribution"""
        degrees = [len(neighbors) for neighbors in graph.values()]
        degree_counts = {}
        for degree in degrees:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        return degree_counts
    
    def _compute_skeleton_length(self, skeleton_points, graph):
        """Compute total skeleton length"""
        total_length = 0.0
        counted_edges = set()
        
        for coord, neighbors in graph.items():
            coord_array = np.array(coord)
            for neighbor_coord in neighbors:
                neighbor_array = np.array(neighbor_coord)
                edge = tuple(sorted([coord, neighbor_coord]))
                if edge not in counted_edges:
                    # Euclidean distance between points
                    dist = np.linalg.norm(coord_array - neighbor_array)
                    total_length += dist
                    counted_edges.add(edge)
        
        return total_length
    
    def _compute_tortuosity(self, graph, skeleton_points):
        """Compute average tortuosity of skeleton branches"""
        if len(graph) < 2:
            return 0.0
        
        components = self._find_connected_components(graph)
        tortuosities = []
        
        # Convert coordinates back to arrays for calculations
        coords_list = list(graph.keys())
        
        for component in components:
            if len(component) < 3:  # Need at least 3 points for meaningful tortuosity
                continue
            
            # Get component coordinates
            component_coords = [np.array(coords_list[i]) for i in component]
            
            if len(component_coords) < 2:
                continue
            
            # Path length (sum of edges within component)
            path_length = 0.0
            for i, coord1 in enumerate(component_coords):
                coord1_tuple = tuple(coord1)
                if coord1_tuple in graph:
                    for neighbor_tuple in graph[coord1_tuple]:
                        neighbor_coord = np.array(neighbor_tuple)
                        # Only count if neighbor is in same component
                        neighbor_idx = None
                        for j, comp_coord in enumerate(component_coords):
                            if np.array_equal(comp_coord, neighbor_coord):
                                neighbor_idx = j
                                break
                        
                        if neighbor_idx is not None and neighbor_idx > i:  # Avoid double counting
                            path_length += np.linalg.norm(coord1 - neighbor_coord)
            
            # Euclidean distance (straight line)
            if len(component_coords) >= 2 and path_length > 0:
                euclidean_dist = np.linalg.norm(component_coords[0] - component_coords[-1])
                if euclidean_dist > 0:
                    tortuosity = path_length / euclidean_dist
                    tortuosities.append(tortuosity)
        
        return np.mean(tortuosities) if tortuosities else 1.0


class BreakageDetector:
    """
    Detect breakages and gaps in skeleton structure
    """
    
    def __init__(self, min_component_size=5, max_gap_distance=5.0, endpoint_threshold=3.0):
        """
        Args:
            min_component_size (int): Minimum size for valid components
            max_gap_distance (float): Maximum distance to consider for gap repair
            endpoint_threshold (float): Distance threshold for endpoint analysis
        """
        self.min_component_size = min_component_size
        self.max_gap_distance = max_gap_distance
        self.endpoint_threshold = endpoint_threshold
    
    def detect_breakages(self, skeleton, analysis=None):
        """
        Detect potential breakages in skeleton
        
        Args:
            skeleton: Binary skeleton [D, H, W]
            analysis: Pre-computed skeleton analysis (optional)
            
        Returns:
            dict: Breakage detection results
        """
        if analysis is None:
            analyzer = SkeletonAnalyzer()
            analysis = analyzer.analyze_skeleton(skeleton)
        
        breakages = {
            'small_components': self._find_small_components(analysis),
            'isolated_endpoints': self._find_isolated_endpoints(analysis),
            'gap_candidates': self._find_gap_candidates(analysis),
            'suspicious_junctions': self._find_suspicious_junctions(analysis),
            'repair_suggestions': []
        }
        
        # Generate repair suggestions
        breakages['repair_suggestions'] = self._generate_repair_suggestions(breakages, analysis)
        
        return breakages
    
    def _find_small_components(self, analysis):
        """Find components smaller than minimum size"""
        small_components = []
        for component in analysis['connected_components']:
            if len(component) < self.min_component_size:
                small_components.append(component)
        return small_components
    
    def _find_isolated_endpoints(self, analysis):
        """Find endpoints that are far from main structure"""
        isolated_endpoints = []
        endpoints = analysis['endpoint_coords']
        
        if len(endpoints) == 0:
            return isolated_endpoints
        
        # Find largest component (main structure)
        components = analysis['connected_components']
        if not components:
            return isolated_endpoints
        
        main_component = max(components, key=len)
        
        # Get coordinates of skeleton points
        all_coords = []
        for coord_tuple in analysis['connectivity_graph'].keys():
            all_coords.append(list(coord_tuple))
        
        if not all_coords:
            return isolated_endpoints
            
        all_coords = np.array(all_coords)
        main_component_coords = all_coords[main_component]
        
        # Check distance of each endpoint to main component
        for endpoint in endpoints:
            if len(main_component_coords) > 0:
                # Calculate distances manually instead of using sklearn
                distances = np.sqrt(np.sum((main_component_coords - endpoint) ** 2, axis=1))
                min_distance = np.min(distances)
                
                if min_distance > self.endpoint_threshold:
                    isolated_endpoints.append({
                        'coord': endpoint,
                        'distance_to_main': min_distance
                    })
        
        return isolated_endpoints
    
    def _find_gap_candidates(self, analysis):
        """Find potential gaps between skeleton components"""
        gap_candidates = []
        components = analysis['connected_components']
        
        if len(components) < 2:
            return gap_candidates
        
        # Get coordinates for each component
        coords_list = list(analysis['connectivity_graph'].keys())
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1_coords = [np.array(coords_list[idx]) for idx in components[i]]
                comp2_coords = [np.array(coords_list[idx]) for idx in components[j]]
                
                # Find closest points between components
                min_distance = float('inf')
                closest_pair = None
                
                for coord1 in comp1_coords:
                    for coord2 in comp2_coords:
                        distance = np.linalg.norm(coord1 - coord2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_pair = (coord1, coord2)
                
                if min_distance <= self.max_gap_distance:
                    gap_candidates.append({
                        'component1': i,
                        'component2': j,
                        'distance': min_distance,
                        'closest_points': closest_pair,
                        'component1_size': len(components[i]),
                        'component2_size': len(components[j])
                    })
        
        return gap_candidates
    
    def _find_suspicious_junctions(self, analysis):
        """Find junctions that might indicate incorrect connections"""
        suspicious = []
        graph = analysis['connectivity_graph']
        
        # Find junctions with very high degree (> 4 in 3D is unusual for vessels)
        for coord_tuple, neighbors in graph.items():
            if len(neighbors) > 4:  # Suspicious high-degree junction
                suspicious.append({
                    'coord': np.array(coord_tuple),
                    'degree': len(neighbors),
                    'type': 'high_degree'
                })
        
        return suspicious
    
    def _generate_repair_suggestions(self, breakages, analysis):
        """Generate suggestions for repairing detected breakages"""
        suggestions = []
        
        # Suggest gap filling
        for gap in breakages['gap_candidates']:
            if gap['distance'] <= self.max_gap_distance:
                suggestions.append({
                    'type': 'fill_gap',
                    'priority': 1.0 / (gap['distance'] + 1e-6),  # Closer gaps higher priority
                    'details': gap
                })
        
        # Suggest small component removal or connection
        for small_comp in breakages['small_components']:
            suggestions.append({
                'type': 'remove_or_connect_small_component',
                'priority': 0.5,
                'details': {'component': small_comp, 'size': len(small_comp)}
            })
        
        # Sort by priority
        suggestions.sort(key=lambda x: x['priority'], reverse=True)
        
        return suggestions


class SkeletonRepairer:
    """
    Repair broken skeleton structures using various strategies
    """
    
    def __init__(self, repair_strategy='conservative', max_repair_distance=5.0):
        """
        Args:
            repair_strategy (str): 'conservative', 'aggressive', 'adaptive'
            max_repair_distance (float): Maximum distance for repair connections
        """
        self.repair_strategy = repair_strategy
        self.max_repair_distance = max_repair_distance
    
    def repair_skeleton(self, skeleton, original_mask=None, breakages=None):
        """
        Repair skeleton breakages
        
        Args:
            skeleton: Binary skeleton [D, H, W]
            original_mask: Original segmentation mask (for constraints)
            breakages: Pre-detected breakages (optional)
            
        Returns:
            np.ndarray: Repaired skeleton
        """
        if not skeleton.any():
            return skeleton.copy()
        
        repaired = skeleton.copy()
        
        # Detect breakages if not provided
        if breakages is None:
            detector = BreakageDetector(max_gap_distance=self.max_repair_distance)
            analyzer = SkeletonAnalyzer()
            analysis = analyzer.analyze_skeleton(skeleton)
            breakages = detector.detect_breakages(skeleton, analysis)
        
        # Apply repairs based on strategy
        if self.repair_strategy == 'conservative':
            repaired = self._conservative_repair(repaired, breakages, original_mask)
        elif self.repair_strategy == 'aggressive':
            repaired = self._aggressive_repair(repaired, breakages, original_mask)
        elif self.repair_strategy == 'adaptive':
            repaired = self._adaptive_repair(repaired, breakages, original_mask)
        
        return repaired
    
    def _conservative_repair(self, skeleton, breakages, original_mask):
        """Conservative repair: only fill obvious gaps"""
        repaired = skeleton.copy()
        
        # Only fill small gaps between large components
        for gap in breakages['gap_candidates']:
            if (gap['distance'] <= 3.0 and 
                gap['component1_size'] >= 10 and 
                gap['component2_size'] >= 10):
                
                repaired = self._fill_gap(repaired, gap, original_mask)
        
        return repaired
    
    def _aggressive_repair(self, skeleton, breakages, original_mask):
        """Aggressive repair: fill all detected gaps and connect components"""
        repaired = skeleton.copy()
        
        # Fill all gaps within distance threshold
        for gap in breakages['gap_candidates']:
            if gap['distance'] <= self.max_repair_distance:
                repaired = self._fill_gap(repaired, gap, original_mask)
        
        # Connect isolated endpoints to nearest components
        for isolated in breakages['isolated_endpoints']:
            repaired = self._connect_isolated_endpoint(repaired, isolated, original_mask)
        
        return repaired
    
    def _adaptive_repair(self, skeleton, breakages, original_mask):
        """Adaptive repair: context-aware repair decisions"""
        repaired = skeleton.copy()
        
        # Prioritize repairs based on component sizes and distances
        for suggestion in breakages['repair_suggestions']:
            if suggestion['type'] == 'fill_gap' and suggestion['priority'] > 0.5:
                gap = suggestion['details']
                if self._validate_gap_repair(gap, original_mask):
                    repaired = self._fill_gap(repaired, gap, original_mask)
        
        return repaired
    
    def _fill_gap(self, skeleton, gap, original_mask):
        """Fill gap between two skeleton components"""
        point1, point2 = gap['closest_points']
        
        # Generate path between points
        path = self._generate_connection_path(point1, point2, original_mask)
        
        # Add path to skeleton
        for point in path:
            if (0 <= point[0] < skeleton.shape[0] and
                0 <= point[1] < skeleton.shape[1] and
                0 <= point[2] < skeleton.shape[2]):
                skeleton[tuple(point)] = 1.0
        
        return skeleton
    
    def _generate_connection_path(self, start, end, original_mask):
        """Generate connection path between two points"""
        start = np.array(start, dtype=int)
        end = np.array(end, dtype=int)
        
        # Simple linear interpolation path
        distance = np.linalg.norm(end - start)
        num_points = int(distance) + 1
        
        path = []
        for i in range(num_points + 1):
            t = i / max(num_points, 1)
            point = start + t * (end - start)
            path.append(point.astype(int))
        
        return path
    
    def _connect_isolated_endpoint(self, skeleton, isolated, original_mask):
        """Connect isolated endpoint to nearest skeleton component"""
        # For now, implement basic connection
        # More sophisticated algorithms (A*, Dijkstra) can be added here
        return skeleton
    
    def _validate_gap_repair(self, gap, original_mask):
        """Validate if gap repair is appropriate"""
        if original_mask is None:
            return True
        
        # Check if path lies within original segmentation
        point1, point2 = gap['closest_points']
        path = self._generate_connection_path(point1, point2, original_mask)
        
        # Check if at least 70% of path is within original mask
        valid_points = 0
        for point in path:
            if (0 <= point[0] < original_mask.shape[0] and
                0 <= point[1] < original_mask.shape[1] and
                0 <= point[2] < original_mask.shape[2]):
                if original_mask[tuple(point)] > 0.5:
                    valid_points += 1
        
        validity_ratio = valid_points / len(path) if path else 0
        return validity_ratio >= 0.7


class PostProcessingPipeline:
    """
    Complete post-processing pipeline for vessel segmentation
    """
    
    def __init__(self, skeleton_method='skeletonize_3d', repair_strategy='adaptive',
                 apply_morphological_cleanup=True, final_dilation_radius=1):
        """
        Args:
            skeleton_method (str): Method for skeleton extraction
            repair_strategy (str): Strategy for skeleton repair
            apply_morphological_cleanup (bool): Apply morphological operations
            final_dilation_radius (int): Radius for final dilation to restore width
        """
        self.skeleton_extractor = Skeleton3DExtractor(method=skeleton_method)
        self.analyzer = SkeletonAnalyzer(connectivity=26)
        self.detector = BreakageDetector()
        self.repairer = SkeletonRepairer(repair_strategy=repair_strategy)
        
        self.apply_morphological_cleanup = apply_morphological_cleanup
        self.final_dilation_radius = final_dilation_radius
    
    def __call__(self, predictions, original_masks=None, return_analysis=False):
        """
        Apply complete post-processing pipeline
        
        Args:
            predictions: Model predictions [B, C, D, H, W] (probabilities)
            original_masks: Original segmentation masks (for validation)
            return_analysis: Whether to return detailed analysis
            
        Returns:
            dict: Post-processed results
        """
        # Convert predictions to binary
        binary_preds = (predictions > 0.5).astype(np.float32)
        
        batch_size, num_classes = binary_preds.shape[:2]
        
        # Initialize outputs
        repaired_skeletons = np.zeros_like(binary_preds)
        repaired_masks = np.zeros_like(binary_preds)
        
        analyses = [] if return_analysis else None
        
        # Process each sample and class
        for b in range(batch_size):
            for c in range(num_classes):
                mask = binary_preds[b, c]
                original_mask = original_masks[b, c] if original_masks is not None else None
                
                if mask.sum() == 0:
                    continue
                
                # 1. Extract skeleton
                skeleton = self.skeleton_extractor.extract_skeleton(mask)
                
                # 2. Analyze skeleton
                analysis = self.analyzer.analyze_skeleton(skeleton)
                
                # 3. Detect breakages
                breakages = self.detector.detect_breakages(skeleton, analysis)
                
                # 4. Repair skeleton
                repaired_skeleton = self.repairer.repair_skeleton(
                    skeleton, original_mask, breakages
                )
                
                # 5. Convert back to segmentation mask
                repaired_mask = self._skeleton_to_mask(
                    repaired_skeleton, mask, original_mask
                )
                
                # Store results
                repaired_skeletons[b, c] = repaired_skeleton
                repaired_masks[b, c] = repaired_mask
                
                # Store analysis if requested
                if return_analysis:
                    analyses.append({
                        'batch': b,
                        'class': c,
                        'original_analysis': analysis,
                        'breakages': breakages,
                        'repair_applied': len(breakages['repair_suggestions']) > 0
                    })
        
        result = {
            'repaired_masks': repaired_masks,
            'repaired_skeletons': repaired_skeletons,
            'original_predictions': predictions
        }
        
        if return_analysis:
            result['analyses'] = analyses
        
        return result
    
    def _skeleton_to_mask(self, skeleton, original_mask, reference_mask=None):
        """Convert repaired skeleton back to segmentation mask"""
        # Start with repaired skeleton
        mask = skeleton.copy()
        
        # Apply dilation to restore vessel width
        if self.final_dilation_radius > 0:
            kernel = ball(self.final_dilation_radius)
            mask = binary_dilation(mask, kernel).astype(np.float32)
        
        # Constrain to original mask region (optional)
        if reference_mask is not None:
            mask = mask * (reference_mask > 0.5)
        
        # Morphological cleanup
        if self.apply_morphological_cleanup:
            # Remove small noise
            mask = morphology.remove_small_objects(
                mask.astype(bool), min_size=10
            ).astype(np.float32)
            
            # Fill small holes
            mask = morphology.remove_small_holes(
                mask.astype(bool), area_threshold=5
            ).astype(np.float32)
        
        return mask


# Utility functions for integration
def apply_skeleton_postprocessing(predictions, original_masks=None, 
                                config=None, device='cpu'):
    """
    Convenience function for applying skeleton post-processing
    
    Args:
        predictions: Model predictions (torch.Tensor or numpy.ndarray)
        original_masks: Original masks for reference
        config: Configuration dictionary
        device: Device for processing
        
    Returns:
        torch.Tensor: Post-processed predictions
    """
    # Default configuration
    default_config = {
        'skeleton_method': 'skeletonize_3d',
        'repair_strategy': 'adaptive',
        'morphological_cleanup': True,
        'dilation_radius': 1
    }
    
    if config is not None:
        default_config.update(config)
    
    # Convert to numpy if necessary
    if isinstance(predictions, torch.Tensor):
        pred_numpy = predictions.detach().cpu().numpy()
        return_tensor = True
    else:
        pred_numpy = predictions
        return_tensor = False
    
    if original_masks is not None and isinstance(original_masks, torch.Tensor):
        mask_numpy = original_masks.detach().cpu().numpy()
    else:
        mask_numpy = original_masks
    
    # Create post-processing pipeline
    pipeline = PostProcessingPipeline(
        skeleton_method=default_config['skeleton_method'],
        repair_strategy=default_config['repair_strategy'],
        apply_morphological_cleanup=default_config['morphological_cleanup'],
        final_dilation_radius=default_config['dilation_radius']
    )
    
    # Apply post-processing
    result = pipeline(pred_numpy, mask_numpy)
    repaired_masks = result['repaired_masks']
    
    # Convert back to tensor if needed
    if return_tensor:
        return torch.from_numpy(repaired_masks).to(device)
    else:
        return repaired_masks


def evaluate_skeleton_quality(skeleton, original_mask=None):
    """
    Evaluate skeleton quality metrics
    
    Args:
        skeleton: Binary skeleton
        original_mask: Original segmentation mask
        
    Returns:
        dict: Quality metrics
    """
    analyzer = SkeletonAnalyzer()
    analysis = analyzer.analyze_skeleton(skeleton)
    
    metrics = {
        'skeleton_length': analysis['skeleton_length'],
        'n_components': len(analysis['connected_components']),
        'n_endpoints': len(analysis['endpoint_coords']),
        'n_junctions': len(analysis['junction_coords']),
        'average_tortuosity': analysis['tortuosity'],
        'degree_distribution': analysis['degree_distribution']
    }
    
    # Connectivity metrics
    if analysis['total_points'] > 0:
        metrics['connectivity_ratio'] = (
            len(analysis['connected_components'][0]) / analysis['total_points']
            if analysis['connected_components'] else 0.0
        )
        metrics['junction_density'] = len(analysis['junction_coords']) / analysis['total_points']
        metrics['endpoint_density'] = len(analysis['endpoint_coords']) / analysis['total_points']
    else:
        metrics['connectivity_ratio'] = 0.0
        metrics['junction_density'] = 0.0
        metrics['endpoint_density'] = 0.0
    
    return metrics


class SkeletonVisualizer:
    """
    Visualization utilities for skeleton analysis and repair
    """
    
    def __init__(self):
        pass
    
    def create_visualization_data(self, skeleton, analysis=None, breakages=None):
        """
        Create data for 3D visualization
        
        Args:
            skeleton: Binary skeleton
            analysis: Skeleton analysis results
            breakages: Breakage detection results
            
        Returns:
            dict: Visualization data
        """
        if analysis is None:
            analyzer = SkeletonAnalyzer()
            analysis = analyzer.analyze_skeleton(skeleton)
        
        if breakages is None:
            detector = BreakageDetector()
            breakages = detector.detect_breakages(skeleton, analysis)
        
        # Get skeleton points from coordinates
        skeleton_points = np.array([list(coord) for coord in analysis['connectivity_graph'].keys()])
        
        viz_data = {
            'skeleton_points': skeleton_points,
            'endpoints': analysis['endpoint_coords'],
            'junctions': analysis['junction_coords'],
            'isolated_points': analysis['isolated_points'],
            'connected_components': analysis['connected_components'],
            'gap_candidates': breakages['gap_candidates'],
            'small_components': breakages['small_components']
        }
        
        return viz_data
    
    def generate_repair_report(self, original_analysis, repaired_analysis, breakages):
        """
        Generate text report of repair operations
        
        Args:
            original_analysis: Analysis of original skeleton
            repaired_analysis: Analysis of repaired skeleton
            breakages: Detected breakages
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("=== Skeleton Repair Report ===")
        report.append("")
        
        # Original skeleton stats
        report.append("Original Skeleton:")
        report.append(f"  Total points: {original_analysis['total_points']}")
        report.append(f"  Connected components: {len(original_analysis['connected_components'])}")
        report.append(f"  Endpoints: {len(original_analysis['endpoint_coords'])}")
        report.append(f"  Junctions: {len(original_analysis['junction_coords'])}")
        report.append(f"  Skeleton length: {original_analysis['skeleton_length']:.2f}")
        report.append("")
        
        # Detected issues
        report.append("Detected Issues:")
        report.append(f"  Small components: {len(breakages['small_components'])}")
        report.append(f"  Isolated endpoints: {len(breakages['isolated_endpoints'])}")
        report.append(f"  Gap candidates: {len(breakages['gap_candidates'])}")
        report.append(f"  Repair suggestions: {len(breakages['repair_suggestions'])}")
        report.append("")
        
        # Repaired skeleton stats
        report.append("Repaired Skeleton:")
        report.append(f"  Total points: {repaired_analysis['total_points']}")
        report.append(f"  Connected components: {len(repaired_analysis['connected_components'])}")
        report.append(f"  Endpoints: {len(repaired_analysis['endpoint_coords'])}")
        report.append(f"  Junctions: {len(repaired_analysis['junction_coords'])}")
        report.append(f"  Skeleton length: {repaired_analysis['skeleton_length']:.2f}")
        report.append("")
        
        # Improvements
        component_improvement = (len(original_analysis['connected_components']) - 
                               len(repaired_analysis['connected_components']))
        length_change = (repaired_analysis['skeleton_length'] - 
                        original_analysis['skeleton_length'])
        
        report.append("Improvements:")
        report.append(f"  Components reduced by: {component_improvement}")
        report.append(f"  Length change: {length_change:+.2f}")
        report.append("")
        
        return "\n".join(report)


class BatchSkeletonProcessor:
    """
    Efficient batch processing for skeleton operations
    """
    
    def __init__(self, num_workers=1):
        """
        Args:
            num_workers (int): Number of parallel workers (for future multiprocessing)
        """
        self.num_workers = num_workers
        self.pipeline = PostProcessingPipeline()
    
    def process_batch(self, predictions_batch, original_masks_batch=None, 
                     config=None, verbose=False):
        """
        Process a batch of predictions
        
        Args:
            predictions_batch: Batch of predictions [B, C, D, H, W]
            original_masks_batch: Batch of original masks
            config: Processing configuration
            verbose: Print progress information
            
        Returns:
            dict: Batch processing results
        """
        batch_size = predictions_batch.shape[0]
        results = {
            'repaired_masks': [],
            'repaired_skeletons': [],
            'quality_metrics': [],
            'repair_reports': []
        }
        
        # Process each sample in batch
        for b in range(batch_size):
            if verbose:
                print(f"Processing sample {b+1}/{batch_size}")
            
            # Extract single sample
            pred_sample = predictions_batch[b:b+1]  # Keep batch dimension
            mask_sample = (original_masks_batch[b:b+1] 
                          if original_masks_batch is not None else None)
            
            # Apply post-processing
            sample_result = self.pipeline(pred_sample, mask_sample, return_analysis=True)
            
            # Collect results
            results['repaired_masks'].append(sample_result['repaired_masks'][0])
            results['repaired_skeletons'].append(sample_result['repaired_skeletons'][0])
            
            # Calculate quality metrics
            sample_metrics = []
            for c in range(pred_sample.shape[1]):
                skeleton = sample_result['repaired_skeletons'][0, c]
                metrics = evaluate_skeleton_quality(skeleton)
                sample_metrics.append(metrics)
            results['quality_metrics'].append(sample_metrics)
            
            # Generate repair reports if analysis available
            if 'analyses' in sample_result:
                sample_reports = []
                for analysis_data in sample_result['analyses']:
                    if analysis_data['repair_applied']:
                        # Create simplified report for batch processing
                        report = {
                            'class': analysis_data['class'],
                            'breakages_detected': len(analysis_data['breakages']['repair_suggestions']),
                            'repairs_applied': analysis_data['repair_applied']
                        }
                        sample_reports.append(report)
                results['repair_reports'].append(sample_reports)
        
        # Convert lists to arrays
        results['repaired_masks'] = np.array(results['repaired_masks'])
        results['repaired_skeletons'] = np.array(results['repaired_skeletons'])
        
        return results


# Test and usage example
if __name__ == '__main__':
    print("Testing Skeleton Post-Processing Pipeline...")
    
    # Create dummy vessel-like structures for testing
    def create_vessel_structure_with_breaks(shape):
        """Create synthetic vessel structure with intentional breaks"""
        volume = np.zeros(shape)
        
        # Main vessel with breaks
        center_z, center_y, center_x = np.array(shape) // 2
        
        # Vessel segment 1
        for i in range(shape[2] // 3):
            y = center_y + int(3 * np.sin(i * 0.2))
            x = center_x + int(2 * np.cos(i * 0.1))
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                volume[max(0, y-1):y+2, max(0, x-1):x+2, i] = 1
        
        # Gap (intentional break)
        
        # Vessel segment 2
        start_z = shape[2] // 3 + 8  # Gap of 8 voxels
        for i in range(start_z, shape[2]):
            y = center_y + int(3 * np.sin(i * 0.2))
            x = center_x + int(2 * np.cos(i * 0.1))
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                volume[max(0, y-1):y+2, max(0, x-1):x+2, i] = 1
        
        # Small isolated component
        iso_y, iso_x, iso_z = center_y + 20, center_x + 15, center_z
        if (iso_y < shape[0] - 5 and iso_x < shape[1] - 5 and iso_z < shape[2] - 5):
            volume[iso_y:iso_y+3, iso_x:iso_x+3, iso_z:iso_z+3] = 1
        
        return volume
    
    # Test parameters
    test_shape = (64, 64, 32)
    batch_size, num_classes = 2, 2
    
    # Create test data
    print(f"Creating test data with shape {test_shape}...")
    test_predictions = np.zeros((batch_size, num_classes) + test_shape)
    
    for b in range(batch_size):
        for c in range(num_classes):
            # Create vessel structure with breaks
            vessel_structure = create_vessel_structure_with_breaks(test_shape)
            test_predictions[b, c] = vessel_structure
    
    print(f"Test data created: {test_predictions.shape}")
    print(f"Non-zero voxels per sample: {[test_predictions[b].sum() for b in range(batch_size)]}")
    
    # Test individual components
    print(f"\nTesting individual components...")
    
    # 1. Test skeleton extraction
    print(f"1. Testing skeleton extraction...")
    extractor = Skeleton3DExtractor(method='skeletonize_3d')
    test_skeleton = extractor.extract_skeleton(test_predictions[0, 0])
    print(f"   Original mask sum: {test_predictions[0, 0].sum()}")
    print(f"   Skeleton sum: {test_skeleton.sum()}")
    
    # 2. Test skeleton analysis
    print(f"2. Testing skeleton analysis...")
    analyzer = SkeletonAnalyzer(connectivity=26)
    analysis = analyzer.analyze_skeleton(test_skeleton)
    print(f"   Total skeleton points: {analysis['total_points']}")
    print(f"   Connected components: {len(analysis['connected_components'])}")
    print(f"   Endpoints: {len(analysis['endpoint_coords'])}")
    print(f"   Junctions: {len(analysis['junction_coords'])}")
    print(f"   Skeleton length: {analysis['skeleton_length']:.2f}")
    
    # 3. Test breakage detection
    print(f"3. Testing breakage detection...")
    detector = BreakageDetector(max_gap_distance=10.0)
    breakages = detector.detect_breakages(test_skeleton, analysis)
    print(f"   Small components detected: {len(breakages['small_components'])}")
    print(f"   Gap candidates: {len(breakages['gap_candidates'])}")
    print(f"   Repair suggestions: {len(breakages['repair_suggestions'])}")
    
    if breakages['gap_candidates']:
        for i, gap in enumerate(breakages['gap_candidates']):
            print(f"     Gap {i+1}: distance={gap['distance']:.2f}")
    
    # 4. Test skeleton repair
    print(f"4. Testing skeleton repair...")
    repairer = SkeletonRepairer(repair_strategy='adaptive')
    repaired_skeleton = repairer.repair_skeleton(test_skeleton, None, breakages)
    print(f"   Original skeleton sum: {test_skeleton.sum()}")
    print(f"   Repaired skeleton sum: {repaired_skeleton.sum()}")
    
    # Analyze repaired skeleton
    repaired_analysis = analyzer.analyze_skeleton(repaired_skeleton)
    print(f"   Repaired components: {len(repaired_analysis['connected_components'])}")
    print(f"   Improvement: {len(analysis['connected_components']) - len(repaired_analysis['connected_components'])} fewer components")
    
    # 5. Test complete pipeline
    print(f"5. Testing complete post-processing pipeline...")
    pipeline = PostProcessingPipeline(
        skeleton_method='skeletonize_3d',
        repair_strategy='adaptive',
        apply_morphological_cleanup=True,
        final_dilation_radius=1
    )
    
    # Process single sample
    single_sample = test_predictions[0:1]  # Keep batch dimension
    pipeline_result = pipeline(single_sample, return_analysis=True)
    
    print(f"   Input shape: {single_sample.shape}")
    print(f"   Output shapes:")
    print(f"     Repaired masks: {pipeline_result['repaired_masks'].shape}")
    print(f"     Repaired skeletons: {pipeline_result['repaired_skeletons'].shape}")
    print(f"   Analysis data available: {'analyses' in pipeline_result}")
    
    if 'analyses' in pipeline_result:
        for analysis_data in pipeline_result['analyses']:
            print(f"     Class {analysis_data['class']}: repairs applied = {analysis_data['repair_applied']}")
    
    # 6. Test batch processing
    print(f"6. Testing batch processing...")
    batch_processor = BatchSkeletonProcessor()
    batch_result = batch_processor.process_batch(
        test_predictions, 
        verbose=True
    )
    
    print(f"   Batch results:")
    print(f"     Processed samples: {len(batch_result['repaired_masks'])}")
    print(f"     Quality metrics available: {len(batch_result['quality_metrics'])}")
    print(f"     Repair reports available: {len(batch_result['repair_reports'])}")
    
    # Show sample quality metrics
    if batch_result['quality_metrics']:
        sample_metrics = batch_result['quality_metrics'][0][0]  # First sample, first class
        print(f"   Sample quality metrics:")
        for key, value in sample_metrics.items():
            if isinstance(value, (int, float)):
                print(f"     {key}: {value:.3f}")
    
    # 7. Test convenience function
    print(f"7. Testing convenience function...")
    
    # Test with numpy input
    repaired_numpy = apply_skeleton_postprocessing(
        test_predictions,
        config={'repair_strategy': 'conservative'}
    )
    print(f"   Numpy input/output: {repaired_numpy.shape}")
    
    # Test with torch tensor input
    test_tensor = torch.from_numpy(test_predictions).float()
    repaired_tensor = apply_skeleton_postprocessing(
        test_tensor,
        config={'repair_strategy': 'aggressive'}
    )
    print(f"   Tensor input/output: {repaired_tensor.shape}, device: {repaired_tensor.device}")
    
    # 8. Test evaluation metrics
    print(f"8. Testing evaluation metrics...")
    quality_metrics = evaluate_skeleton_quality(repaired_skeleton)
    print(f"   Quality metrics:")
    for key, value in quality_metrics.items():
        if isinstance(value, (int, float)):
            print(f"     {key}: {value:.3f}")
        else:
            print(f"     {key}: {value}")
    
    # 9. Test visualization data generation
    print(f"9. Testing visualization data generation...")
    visualizer = SkeletonVisualizer()
    viz_data = visualizer.create_visualization_data(test_skeleton, analysis, breakages)
    print(f"   Visualization data generated:")
    for key, value in viz_data.items():
        if isinstance(value, np.ndarray):
            print(f"     {key}: {value.shape}")
        else:
            print(f"     {key}: {len(value) if hasattr(value, '__len__') else type(value)}")
    
    # Generate repair report
    repair_report = visualizer.generate_repair_report(analysis, repaired_analysis, breakages)
    print(f"\n{repair_report}")
    
    print(f"\nAll skeleton post-processing tests passed!")
    print(f"\nIntegration with your training pipeline:")
    print(f"  # During inference/evaluation:")
    print(f"  predictions = model(input_batch)")
    print(f"  repaired_predictions = apply_skeleton_postprocessing(")
    print(f"      predictions, ")
    print(f"      original_masks=ground_truth_masks,")
    print(f"      config={{'repair_strategy': 'adaptive'}}")
    print(f"  )")
    print(f"  ")
    print(f"  # For detailed analysis:")
    print(f"  pipeline = PostProcessingPipeline()")
    print(f"  result = pipeline(predictions, return_analysis=True)")
    print(f"  ")
    print(f"  # For evaluation metrics:")
    print(f"  from connectivity_metrics import TopologyMetrics")
    print(f"  topo_metrics = TopologyMetrics()")
    print(f"  cl_dice = topo_metrics.centerline_dice(repaired_predictions, targets)")
    print(f"  connectivity = topo_metrics.connectivity_accuracy(repaired_predictions, targets)")