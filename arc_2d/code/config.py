"""
Configuration file for ARC-based UAV path planning
Modify parameters here to change training/environment settings
"""

import numpy as np
# ============================================================
# Helper Functions
# ============================================================

def make_triangle(cx, cy, size, angle_offset=0.0):
    """Create triangle obstacle vertices"""
    angles = np.deg2rad([90, 210, 330]) + angle_offset
    return [(cx + size * np.cos(a), cy + size * np.sin(a)) for a in angles]

# Environment Parameters
ENV_CONFIG = {
    'width': 500,  # Environment width in meters
    'height': 500,  # Environment height in meters
    'uav_speed': 3.0,  # UAV speed in m/s
    'target_speed': 0.8,  # Target speed in m/s (0 for stationary)
    'sampling_interval': 1.0,  # Time step in seconds
    'max_steps': 500,  # Maximum steps per episode
    'num_sensors': 9,  # Number of distance sensors
    'sensor_range': 50.0,  # Maximum sensor range in meters
    'd_min_obstacle': 5.0,  # Minimum distance to obstacle (collision threshold)
    'd_min_target': 10.0,  # Target tracking distance threshold
    'num_uavs': 5,  # Number of UAVs exploring simultaneously
    
}

# Network Parameters
NETWORK_CONFIG = {
    'lstm_hidden_size': 128,
    'lstm_num_layers': 1,
    'actor_hidden_size': 256,
    'critic_hidden_size': 256,
    'history_length': 5,  # T in the paper
}

# Training Parameters
TRAIN_CONFIG = {
    'num_episodes': 300,
    'batch_size': 32,
    'memory_capacity': 4000,
    'learning_rate': 1e-3,
    'gamma': 0.9,  # Discount factor
    'tau': 0.01,  # Soft update parameter (epsilon in paper)
    'exploration_noise': 0.2,  # Variance for exploration
    'action_range': 0.157,  # π/20 radians
}

# Reward Function Parameters
REWARD_CONFIG = {
    'lambda_los': 1.0,  # LOS reward weight
    'sigma_obstacle': -0.1,  # Obstacle penalty weight
    'k_terminal': 100.0,  # Terminal reward/penalty
    'alpha_action': -0.01,  # Action smoothness penalty
}

# Environment Definitions
# Each environment is defined by obstacles and target properties
# Configurations match the paper's Figure 8
ENVIRONMENTS = {
    'env1': {
        'name': 'Barrier-free with stationary target',
        'obstacles': [],
        'target_motion': False,
        'target_start': [250, 250],
        'uav_start_range': [[30, 100], [30, 100]],
    },
    'env2': {
        'name': 'Env2_LargeCircleObstacle',
        'obstacles': [
            {'type': 'circle', 'center': (250, 250), 'radius': 120}
        ],
        'target_motion': False,
        'target_start': (420, 460),
        'uav_start_range': [[30, 100], [30, 100]]
    },
    'env3': {
        'name': 'Complex obstacles with stationary target',
        'obstacles': [
        {'type': 'polygon', 'vertices': [(80,80),(120,80),(120,120),(80,120)]},
        {'type': 'polygon', 'vertices': [(180,80),(220,80),(220,120),(180,120)]},
        {'type': 'polygon', 'vertices': [(280,80),(320,80),(320,120),(280,120)]},
        {'type': 'polygon', 'vertices': [(380,80),(420,80),(420,120),(380,120)]},

        {'type': 'polygon', 'vertices': [(80,180),(120,180),(120,220),(80,220)]},
        {'type': 'polygon', 'vertices': [(180,180),(220,180),(220,220),(180,220)]},
        {'type': 'polygon', 'vertices': [(280,180),(320,180),(320,220),(280,220)]},
        {'type': 'polygon', 'vertices': [(380,180),(420,180),(420,220),(380,220)]},

        {'type': 'polygon', 'vertices': [(80,280),(120,280),(120,320),(80,320)]},
        {'type': 'polygon', 'vertices': [(180,280),(220,280),(220,320),(180,320)]},
        {'type': 'polygon', 'vertices': [(280,280),(320,280),(320,320),(280,320)]},
        {'type': 'polygon', 'vertices': [(380,280),(420,280),(420,320),(380,320)]}
    ],
        'target_motion': False,
        'target_start': [200, 150],
        'uav_start_range': [[50, 50], [100, 200]],
    },
    'env4': {
        'name': 'Barrier-free with moving target',
        'obstacles': [],
        'target_motion': True,
        'target_start': [50, 50],
        'target_velocity': [1, 1],  # Moving diagonally
        'uav_start_range': [[350, 420], [350, 420]],
    },
    'env5': {
        'name': 'Simple obstacle with moving target',
        'obstacles': [
                    {'type': 'circle', 'center': (240, 340), 'radius': 40}
                ],
        'target_motion': True,
        'target_start': (180, 200),
        'target_velocity': (1.5, 1.35),
        'uav_start_range': [(50, 460), (50, 460)],
    },
    'env6': {
        'name': 'Complex obstacles with moving target',
        'obstacles': [
            {'type': 'polygon', 'vertices': [[40, 350], [90, 380], [60, 430], [10, 400]]},
            {'type': 'polygon', 'vertices': [[420, 350], [470, 380], [440, 430], [390, 400]]},
            {'type': 'polygon', 'vertices': [[100, 420], [150, 450], [120, 490], [70, 460]]},
            {'type': 'polygon', 'vertices': [[350, 420], [400, 450], [370, 490], [320, 460]]},
            {'type': 'polygon', 'vertices': [[250, 100], [300, 130], [270, 180], [220, 150]]},
            {'type': 'polygon', 'vertices': [[160, 180], [210, 210], [180, 260], [130, 230]]},
            {'type': 'polygon', 'vertices': [[290, 250], [340, 280], [310, 330], [260, 300]]},
            {'type': 'polygon', 'vertices': [[100, 280], [150, 310], [120, 360], [70, 330]]},
            {'type': 'polygon', 'vertices': [[350, 150], [400, 180], [370, 230], [320, 200]]},
        ],
        'target_motion': True,
        'target_start': [250, 450],
        'target_velocity': [0.0, -0.8],  # Moving upward
        'uav_start_range': [[150, 100], [150, 100]],
    },
    'env7': {
        'name': 'Sparse obstacles with centered stationary target',
        'obstacles': [
            {'type': 'polygon', 'vertices': [[100, 100], [150, 100], [150, 150], [100, 150]]},
            {'type': 'polygon', 'vertices': [[350, 100], [400, 100], [400, 150], [350, 150]]},
            {'type': 'polygon', 'vertices': [[100, 350], [150, 350], [150, 400], [100, 400]]},
        ],
        'target_motion': False,
        'target_start': [250, 250],
        'uav_start_range': [[50, 450], [50, 450]],
    },
    'env8': {
        'name': 'Dense obstacles with centered stationary target',
        'obstacles': [
            {'type': 'polygon', 'vertices': [[80, 80], [120, 80], [120, 150], [80, 150]]},
            {'type': 'polygon', 'vertices': [[380, 80], [420, 80], [420, 150], [380, 150]]},
            {'type': 'polygon', 'vertices': [[80, 350], [120, 350], [120, 420], [80, 420]]},
            {'type': 'polygon', 'vertices': [[380, 350], [420, 350], [420, 420], [380, 420]]},
            {'type': 'circle', 'center': [150, 250], 'radius': 30},
            {'type': 'circle', 'center': [350, 250], 'radius': 30},
            {'type': 'polygon', 'vertices': [[220, 150], [280, 150], [280, 180], [220, 180]]},
            {'type': 'polygon', 'vertices': [[220, 320], [280, 320], [280, 350], [220, 350]]},
        ],
        'target_motion': False,
        'target_start': [250, 250],
        'uav_start_range': [[50, 450], [50, 450]],
    },
    
    'env2_converted': {
    'name': 'Env2_LargeCircleObstacle',
    'obstacles': [
        {
            'type': 'circle',
            'center': (250, 300),
            'radius': 130
        }
    ],
    'target_motion': False,
    'target_start': [425, 465],
    'uav_start_range': [[50, 70], [60, 80]]
},'env3_converted': {
    'name': 'Complex obstacles with stationary target',
    'obstacles': [
        {'type': 'polygon', 'vertices': [(95, 75), (145, 75), (145, 125), (95, 125)]},
        {'type': 'polygon', 'vertices': [(185, 75), (235, 75), (235, 125), (185, 125)]},
        {'type': 'polygon', 'vertices': [(275, 75), (325, 75), (325, 125), (275, 125)]},
        {'type': 'polygon', 'vertices': [(365, 75), (415, 75), (415, 125), (365, 125)]},

        {'type': 'polygon', 'vertices': [(95, 195), (145, 195), (145, 245), (95, 245)]},
        {'type': 'polygon', 'vertices': [(185, 195), (235, 195), (235, 245), (185, 245)]},
        {'type': 'polygon', 'vertices': [(275, 195), (325, 195), (325, 245), (275, 245)]},
        {'type': 'polygon', 'vertices': [(365, 195), (415, 195), (415, 245), (365, 245)]},

        {'type': 'polygon', 'vertices': [(95, 315), (145, 315), (145, 365), (95, 365)]},
        {'type': 'polygon', 'vertices': [(185, 315), (235, 315), (235, 365), (185, 365)]},
        {'type': 'polygon', 'vertices': [(275, 315), (325, 315), (325, 365), (275, 365)]},
        {'type': 'polygon', 'vertices': [(365, 315), (415, 315), (415, 365), (365, 365)]},
    ],
    'target_motion': False,
    'target_start': [190, 150],
    'uav_start_range': [[70, 90], [130, 160]]
},
'env6_converted': {
    'name': 'Complex obstacles with stationary target',
    'obstacles': [
        {'type': 'polygon', 'vertices': [(70, 345), (130, 345), (130, 405), (70, 405)]},
        {'type': 'circle', 'center': (100, 250), 'radius': 30},
        {'type': 'circle', 'center': (100, 110), 'radius': 30},

        {'type': 'circle', 'center': (225, 340), 'radius': 30},
        {'type': 'circle', 'center': (225, 210), 'radius': 30},
        {'type': 'polygon', 'vertices': [(210, 60), (240, 60), (240, 120), (210, 120)]},

        {'type': 'polygon', 'vertices': [(320, 330), (380, 330), (380, 390), (320, 390)]},
        {'type': 'polygon', 'vertices': [(320, 210), (380, 210), (380, 270), (320, 270)]},
        {'type': 'circle', 'center': (350, 100), 'radius': 30},
    ],
    'target_motion': False,
    'target_start': [325, 40],
    'uav_start_range': [[70, 90], [420, 440]]
},
'env_apf_urban': {
    'name': 'Urban District — Research Layout (APF)',

    'obstacles': [
        # --- Buildings (scaled 50x from 0-10 to 0-500) ---
        {'type': 'polygon', 'vertices': [
            (150-30, 300-75), (150+30, 300-75),
            (150+30, 300+75), (150-30, 300+75)
        ]},

        {'type': 'polygon', 'vertices': [
            (400-30, 250-75), (400+30, 250-75),
            (400+30, 250+75), (400-30, 250+75)
        ]},

        # --- Utility blocks ---
        {'type': 'polygon', 'vertices': [
            (250-30, 130-30), (250+30, 130-30),
            (250+30, 130+30), (250-30, 130+30)
        ]},

        {'type': 'polygon', 'vertices': [
            (250-30, 370-30), (250+30, 370-30),
            (250+30, 370+30), (250-30, 370+30)
        ]},

        # --- Triangles (debris) ---
        {'type': 'polygon', 'vertices': make_triangle(150, 110, 32.5, 0.0)},
        {'type': 'polygon', 'vertices': make_triangle(250, 250, 32.5, 0.5)},
        {'type': 'polygon', 'vertices': make_triangle(360, 390, 32.5, 1.0)},

        # --- Circles ---
        {'type': 'circle', 'center': (400, 140), 'radius': 25},

        # --- Outliers ---
        {'type': 'circle', 'center': (40, 460), 'radius': 32.5},
        {'type': 'circle', 'center': (460, 40), 'radius': 32.5},
    ],

    'target_motion': False,

    # Goal ≈ target (scaled to 0-500)
    'target_start': [465, 465],

    # Start ≈ UAV spawn (scaled to 0-500, wide range away from obstacles)
    'uav_start_range': [[30, 100], [30, 100]],
},
'corridor_gauntlet': {
    # Exact copy of the APF "Corridor Gauntlet" layout (static_apf.py, env 1):
    # 2 rectangles (60x150 m), 2 squares (60x60 m), 3 triangles, 3 circles
    # (the two r=32.5 corner circles are the stationary "outliers" of the
    # dynamic suite). This is the shared custom environment for APF and ARC.
    'name': 'Corridor Gauntlet (shared APF/ARC layout)',
    'obstacles': [
        {'type': 'polygon', 'vertices': [(150, 205), (210, 205), (210, 355), (150, 355)]},  # B1
        {'type': 'polygon', 'vertices': [(340, 195), (400, 195), (400, 345), (340, 345)]},  # B2
        {'type': 'polygon', 'vertices': [(200, 150), (260, 150), (260, 210), (200, 210)]},  # B3
        {'type': 'polygon', 'vertices': [(240, 290), (300, 290), (300, 350), (240, 350)]},  # B4
        {'type': 'polygon', 'vertices': make_triangle(185, 155, 32.5, 0.0)},               # V1
        {'type': 'polygon', 'vertices': make_triangle(260, 240, 32.5, 0.5)},               # V2
        {'type': 'polygon', 'vertices': make_triangle(325, 355, 32.5, 1.0)},               # V3
        {'type': 'circle', 'center': (365, 175), 'radius': 25},                            # C1
        {'type': 'circle', 'center': (70, 430), 'radius': 32.5},                           # C2 (outlier)
        {'type': 'circle', 'center': (430, 70), 'radius': 32.5},                           # C3 (outlier)
    ],
    'target_motion': False,
    'target_start': [465, 465],
    'uav_start_range': [[30, 100], [30, 100]],
},
'env_dynamic': {
    'name': 'Urban District — Dynamic Moving Obstacles',
    'obstacles': [
        {'type': 'polygon', 'vertices': [
            (150-30, 300-75), (150+30, 300-75),
            (150+30, 300+75), (150-30, 300+75)
        ]},
        {'type': 'polygon', 'vertices': [
            (400-30, 250-75), (400+30, 250-75),
            (400+30, 250+75), (400-30, 250+75)
        ]},
        {'type': 'polygon', 'vertices': [
            (250-30, 130-30), (250+30, 130-30),
            (250+30, 130+30), (250-30, 130+30)
        ]},
        {'type': 'polygon', 'vertices': [
            (250-30, 370-30), (250+30, 370-30),
            (250+30, 370+30), (250-30, 370+30)
        ]},
        {'type': 'polygon', 'vertices': make_triangle(150, 110, 32.5, 0.0)},
        {'type': 'polygon', 'vertices': make_triangle(250, 250, 32.5, 0.5)},
        {'type': 'polygon', 'vertices': make_triangle(360, 390, 32.5, 1.0)},
        {'type': 'circle', 'center': (400, 140), 'radius': 25},
        {'type': 'circle', 'center': (40, 460), 'radius': 32.5},
        {'type': 'circle', 'center': (460, 40), 'radius': 32.5},
    ],
    'target_motion': False,
    'target_start': [465, 465],
    'uav_start_range': [[30, 100], [30, 100]],
    'dynamic_obstacles': True,
    'obstacle_speed_scale': 15.0
}
}

# ============================================================
# APF-Like Environment Configuration (Scaled to Current Setup)
# ============================================================
APF_REF_BOUNDS = (0.0, 10.0, 0.0, 10.0)
APF_REF_START = (0.7, 0.7)
APF_REF_GOAL = (9.3, 9.3)
APF_REF_DT = 0.05
APF_REF_SPEED_SCALE = 0.30
APF_REF_VELOCITY_SEED = 42

_apf_sx = ENV_CONFIG['width'] / (APF_REF_BOUNDS[1] - APF_REF_BOUNDS[0])
_apf_sy = ENV_CONFIG['height'] / (APF_REF_BOUNDS[3] - APF_REF_BOUNDS[2])
_apf_space_scale = min(_apf_sx, _apf_sy)

def _apf_scale_point(pt):
    return (float(pt[0] * _apf_sx), float(pt[1] * _apf_sy))

def _apf_rect_to_polygon(cx, cy, hw, hh):
    cxs, cys = _apf_scale_point((cx, cy))
    hws = hw * _apf_sx
    hhs = hh * _apf_sy
    return [
        (cxs - hws, cys - hhs),
        (cxs + hws, cys - hhs),
        (cxs + hws, cys + hhs),
        (cxs - hws, cys + hhs),
    ]

def _apf_to_env_obstacle(obs):
    out = {'type': obs['type'], 'outlier': obs.get('outlier', False), 'label': obs.get('label', '')}
    t = obs['type']
    if t in ('rectangle', 'square'):
        cx, cy, hw, hh = obs['params']
        out['type'] = 'polygon'
        out['vertices'] = _apf_rect_to_polygon(cx, cy, hw, hh)
    elif t == 'triangle':
        out['type'] = 'polygon'
        out['vertices'] = [_apf_scale_point(v) for v in obs['params']]
    elif t == 'circle':
        cx, cy, r = obs['params']
        out['type'] = 'circle'
        out['center'] = _apf_scale_point((cx, cy))
        out['radius'] = float(r * _apf_space_scale)
    return out

_APF_REF_OBSTACLES = [
    {'type': 'rectangle', 'params': (3.0, 6.0, 0.60, 1.50), 'label': 'B1-W-Building', 'outlier': False},
    {'type': 'rectangle', 'params': (8.0, 5.0, 0.60, 1.50), 'label': 'B2-E-Building', 'outlier': False},
    {'type': 'square', 'params': (5.0, 2.6, 0.60, 0.60), 'label': 'B3-S-Substation', 'outlier': False},
    {'type': 'square', 'params': (5.0, 7.4, 0.60, 0.60), 'label': 'B4-N-Pumphouse', 'outlier': False},
    {'type': 'triangle', 'params': make_triangle(3.0, 2.2, 0.65, 0.0), 'label': 'V1-SW-Rubble', 'outlier': False},
    {'type': 'triangle', 'params': make_triangle(5.0, 5.0, 0.65, 0.5), 'label': 'V2-C-Debris', 'outlier': False},
    {'type': 'triangle', 'params': make_triangle(7.2, 7.8, 0.65, 1.0), 'label': 'V3-NE-Rubble', 'outlier': False},
    {'type': 'circle', 'params': (8.0, 2.8, 0.50), 'label': 'C2-E-Silo', 'outlier': False},
    {'type': 'circle', 'params': (0.8, 9.2, 0.65), 'label': 'OL1-NW-Mast', 'outlier': True},
    {'type': 'circle', 'params': (9.2, 0.8, 0.65), 'label': 'OL2-SE-Mast', 'outlier': True},
]

_apf_start_scaled = _apf_scale_point(APF_REF_START)
_apf_goal_scaled = _apf_scale_point(APF_REF_GOAL)
_apf_obstacle_substeps = max(1, int(round(ENV_CONFIG['sampling_interval'] / APF_REF_DT)))

ENVIRONMENTS['apf_dynamic'] = {
    'name': 'APF Dynamic Environment (10x10 -> Scaled)',
    'obstacles': [_apf_to_env_obstacle(o) for o in _APF_REF_OBSTACLES],
    'target_motion': False,
    'target_start': [float(_apf_goal_scaled[0]), float(_apf_goal_scaled[1])],
    'uav_start_range': [
        [float(_apf_start_scaled[0]), float(_apf_start_scaled[0])],
        [float(_apf_start_scaled[1]), float(_apf_start_scaled[1])],
    ],
    'dynamic_obstacles': True,
    'obstacle_motion_model': 'apf',
    'obstacle_velocity_seed': APF_REF_VELOCITY_SEED,
    'obstacle_speed_scale': APF_REF_SPEED_SCALE * _apf_space_scale,
    'obstacle_apf_speed_scale': APF_REF_SPEED_SCALE,
    'obstacle_dt': APF_REF_DT,
    'obstacle_substeps': _apf_obstacle_substeps,
    'obstacle_space_scale': _apf_space_scale,
}

# Slower variant for ARC training stability.
ENVIRONMENTS['apf_dynamic_soft'] = {
    **ENVIRONMENTS['apf_dynamic'],
    'name': 'APF Dynamic Environment (Soft Motion)',
    'obstacle_apf_speed_scale': 0.12,
    'obstacle_speed_scale': 0.12 * _apf_space_scale,
}

# ============================================================
# ARC ENVIRONMENT CONFIGURATION (from Jupyter notebook)
# ============================================================
# Helper function for making triangle obstacles
def make_triangle(cx, cy, size, angle_offset=0.0):
    """Create triangle obstacle vertices"""
    angles = np.deg2rad([90, 210, 330]) + angle_offset
    return [(cx + size * np.cos(a), cy + size * np.sin(a)) for a in angles]

# ARC environment configuration
ARC_CONFIG = {
    # Environment bounds (x_min, x_max, y_min, y_max)
    'ENV_BOUNDS': (0.0, 10.0, 0.0, 10.0),
    
    # Start and goal positions
    'START': np.array([0.7, 0.7]),
    'GOAL': np.array([9.3, 9.3]),
    'GOAL_RADIUS': 0.30,
    
    # Obstacles (from notebook)
    'OBSTACLES': [
        {"type": "rectangle", "params": (3.0, 6.0, 0.60, 1.50), "label": "B1"},
        {"type": "rectangle", "params": (8.0, 5.0, 0.60, 1.50), "label": "B2"},
        {"type": "square",    "params": (5.0, 2.6, 0.60, 0.60), "label": "B3"},
        {"type": "square",    "params": (5.0, 7.4, 0.60, 0.60), "label": "B4"},
        {"type": "triangle",  "params": make_triangle(3.0, 2.2, 0.65, 0.0),  "label": "V1"},
        {"type": "triangle",  "params": make_triangle(5.0, 5.0, 0.65, 0.5),  "label": "V2"},
        {"type": "triangle",  "params": make_triangle(7.2, 7.8, 0.65, 1.0),  "label": "V3"},
        {"type": "circle",    "params": (8.0, 2.8, 0.50),  "label": "C2"},
        {"type": "circle",    "params": (0.8, 9.2, 0.65),  "label": "OL1"},
        {"type": "circle",    "params": (9.2, 0.8, 0.65),  "label": "OL2"},
    ],
    
    # Training parameters from notebook
    'USE_TD3': True,
    'NUM_UAVS': 3,
    'N_EPISODES': 1000,
    'PRINT_EVERY': 100,
    'DO_EVAL': True,
    'DO_PLOTS': True,
    
    # Reward function parameters (from notebook patch)
    'R_STEP': -0.02,
    'K_PROGRESS': 4.0,
    'K_ALIGN': 0.3,
    'K_CLEARANCE': 0.8,
    'D_SAFE': 0.80,
    'K_ACTION': 0.01,
    'K_BOUNDARY': 0.20,
    'BOUND_SAFE': 0.50,
    'K_LOS': 0.15,
    'R_GOAL': 250.0,
    'R_COLLISION': -250.0,
}

# Dynamic ARC environment configuration
DYNAMIC_ARC_CONFIG = {
    'ENV_BOUNDS': (0.0, 10.0, 0.0, 10.0),
    'START': np.array([0.7, 0.7]),
    'GOAL': np.array([9.3, 9.3]),
    'GOAL_RADIUS': 0.30,
    
    # Obstacles with outliers defined
    'OBSTACLES': [
        {"type": "rectangle", "params": (3.0, 6.0, 0.60, 1.50), "label": "B1-W-Building",   "outlier": False},
        {"type": "rectangle", "params": (8.0, 5.0, 0.60, 1.50), "label": "B2-E-Building",   "outlier": False},
        {"type": "square",    "params": (5.0, 2.6, 0.60, 0.60), "label": "B3-S-Substation", "outlier": False},
        {"type": "square",    "params": (5.0, 7.4, 0.60, 0.60), "label": "B4-N-Pumphouse",  "outlier": False},
        {"type": "triangle",  "params": make_triangle(3.0, 2.2, 0.65, 0.0), "label": "V1-SW-Rubble",    "outlier": False},
        {"type": "triangle",  "params": make_triangle(5.0, 5.0, 0.65, 0.5), "label": "V2-C-Debris",     "outlier": False},
        {"type": "triangle",  "params": make_triangle(7.2, 7.8, 0.65, 1.0), "label": "V3-NE-Rubble",    "outlier": False},
        {"type": "circle",    "params": (8.0, 2.8, 0.50),       "label": "C2-E-Silo",        "outlier": False},
        {"type": "circle",    "params": (0.8, 9.2, 0.65),       "label": "OL1-NW-Mast",      "outlier": True},
        {"type": "circle",    "params": (9.2, 0.8, 0.65),       "label": "OL2-SE-Mast",      "outlier": True},
    ],
    
    'DYNAMIC_OBSTACLES': True,
    'OBSTACLE_SPEED_SCALE': 0.30,
    
    'USE_TD3': True,
    'NUM_UAVS': 3,
    'N_EPISODES': 1000,
    'PRINT_EVERY': 100,
    'DO_EVAL': True,
    'DO_PLOTS': True,
    
    'R_STEP': -0.02,
    'K_PROGRESS': 4.0,
    'K_ALIGN': 0.3,
    'K_CLEARANCE': 0.8,
    'D_SAFE': 0.80,
    'K_ACTION': 0.01,
    'K_BOUNDARY': 0.20,
    'BOUND_SAFE': 0.50,
    'K_LOS': 0.15,
    'R_GOAL': 250.0,
    'R_COLLISION': -250.0,
}

# Visualization Parameters
VIS_CONFIG = {
    'figure_size': (12, 12),
    'dpi': 120,
    'fps': 15,  # Frames per second for animation
    'trail_length': 80,  # Number of points in UAV trail (longer for better effect)
    'save_dir': 'visualization',
}

# Model Save Parameters
MODEL_CONFIG = {
    'save_dir': 'models',
    'save_interval': 50,  # Save every N episodes
    'best_model_name': 'arc_best.pth',
    'checkpoint_prefix': 'arc_checkpoint',
}

# TensorBoard Parameters
TB_CONFIG = {
    'log_dir': 'runs',
    'flush_secs': 30,
}