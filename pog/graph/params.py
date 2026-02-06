# Parameter file for pog.graph
OFFSET = 0.003  # Offset (z-axis) between two objects (avoids collision between parent and child)
MAX_STABLE_POSES = 5  # Maximum stable poses to be stored for an arbitrary object.
# BULLET_GROUND_OFFSET = [0., 0., -0.0]

# for pog.graph.chromosome
FRICTION_ANGLE_THRESH = 0.1
MAX_INITIAL_TRIES = 100
TF_DIFF_THRESH = 0.01

# {support up: support down}
PairedSurface = {
    "box_aff_pz": "box_aff_nz",
    "box_aff_nz": "box_aff_pz",
    "box_aff_px": "box_aff_nx",
    "box_aff_nx": "box_aff_px",
    "box_aff_py": "box_aff_ny",
    "box_aff_ny": "box_aff_py",
    "cylinder_aff_nz": "cylinder_aff_pz",
    "cylinder_aff_pz": "cylinder_aff_nz",
    "shelf_aff_pz_top": "shelf_aff_nz",
    "shelf_aff_pz_bottom": "shelf_aff_nz",
    "shelf_outer_top": "shelf_outer_bottom",
    "cabinet_inner_bottom": "cabinet_outer_bottom",
    "cabinet_inner_middle": "cabinet_outer_bottom",
    "drawer_inner_bottom": "drawer_outer_bottom",
}

ContainmentSurface = [
    "shelf_aff_pz_bottom",
    "cabinet_inner_bottom",
    "cabinet_inner_middle",
    "drawer_inner_bottom",
]

WALL_THICKNESS = 0.02

COLOR_SOLID = {
    "dark grey": [0.34509804, 0.34509804, 0.34509804, 1.0],
    "light grey": [0.76470588, 0.76470588, 0.76470588, 1.0],
    "red": [0.9254902, 0.10980392, 0.14117647, 1.0],
    "blue": [0.24705882, 0.28235294, 0.8, 0.8],
    "green": [0.05490196, 0.81960784, 0.27058824, 1.0],
    "yellow": [1.0, 0.94901961, 0.0, 1.0],
    "purple": [0.72156863, 0.23921569, 0.72941176, 1.0],
    "brown": [0.7255, 0.4784, 0.3373, 1.0],
    "transparent": [0, 0, 0, 0],
    "red-trans": [0.9254902, 0.10980392, 0.14117647, 1.0],
    "blue-trans": [0.24705882, 0.28235294, 0.8, 1.0],
}

COLOR_IMAGE = {
    "dark grey": [0.34509804, 0.34509804, 0.34509804, 1.0],
    "light grey": [0.76470588, 0.76470588, 0.76470588, 1.0],
    "red": [0.9254902, 0.10980392, 0.14117647, 1.0],
    "blue": [0.24705882, 0.28235294, 0.8, 0.8],
    "green": [0.05490196, 0.81960784, 0.27058824, 1.0],
    "yellow": [1.0, 0.94901961, 0.0, 1.0],
    "purple": [0.72156863, 0.23921569, 0.72941176, 1.0],
    "brown": [0.7255, 0.4784, 0.3373, 1.0],
    "transparent": [0, 0, 0, 0],
    "red-trans": [0.9254902, 0.10980392, 0.14117647, 1.0],
    "blue-trans": [0.24705882, 0.28235294, 0.8, 1.0],
}

COLOR_EXP = {
    "light grey": [0.76470588, 0.76470588, 0.76470588, 1.0],
    "red": [0.9254902, 0.10980392, 0.14117647, 1.0],
    "blue": [0.0, 0.4470, 0.7410, 1.0],
    "orange": [0.8500, 0.3250, 0.0980, 1.0],
    "green": [0.4660, 0.6740, 0.1880, 1.0],
    "yellow": [0.9290, 0.6940, 0.1250, 1.0],
    "purple": [0.4940, 0.1840, 0.5560, 1.0],
    "brown": [0.7255, 0.4784, 0.3373, 1.0],
    "transparent": [0, 0, 0, 0],
    "light blue": [0.3010, 0.7450, 0.9330, 1.0],
    "dark red": [0.6350, 0.0780, 0.1840, 1.0],
    "dark grey": [0.34509804, 0.34509804, 0.34509804, 1.0],
}

COLOR = COLOR_EXP

COLOR_DICT = {
    0: "transparent",
    1: "light grey",
    2: "dark grey",
    3: "yellow",
    4: "orange",
    5: "dark red",
    6: "purple",
    7: "green",
    8: "brown",
    9: "blue",
    99: "brown",
}

BULLET_GROUND_OFFSET = [0.5, -0.5, 0.0]

# For demo on kinova robot arm
GEN3_7DOF_POSES = {
    "home": [-1.57, 0.26, 3.14, -2.27, 0.0, 0.96, 0],
    "retract": [0.0, -0.35, 3.14, -2.54, 0.0, -0.87, 1.57],
    "vertical": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "init": [
        -1.57,
        -0.2934303078299744,
        2.8820947076048085,
        -2.635521573655404,
        -0.45298641397503425,
        0.6932609322127646,
        0.2078563801188996,
    ],
    "goal": [
        -0.7492821764430297,
        0.24286485017039944,
        2.489979400517406,
        -2.4378616584929302,
        0.029921149302870334,
        0.9716196328671798,
        1.6768490635246671,
    ],
    "open": [
        0.325913980174325,
        -0.6582382948454533,
        0.7549810953213919,
        -0.8554660236902418,
        0.029255629242534158,
        -1.9266304517010133,
        2.142120964110152,
    ],
}
