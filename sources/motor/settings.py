layers = {
    "global": {
        "layer_psp": 0.05,
        "layer_ptc": 0.047,
        "layer_pmd": "superelliptic",
        "layer_mod": "even",
    },
    "layers": [
        {
            "type": "start",
            "layer":{
                "layer_pdc": 9,
                "layer_pbw":5.9249,
                "layer_pbh": 8.0710,
                "layer_ppw": 0.547,
                
                # top parameters
                "pattern_tp0": 0.00,
                "pattern_tnn": 2.00,
                "pattern_tmm": 2.00,

                # bottom parameters
                "pattern_bp0": 0.00,
                "pattern_bnn": 2.00,
                "pattern_bmm": 2.00,
                
                "pattern_twist": True,
                "pattern_symmetry": True,
            
                "color": "#f6473e",
            },
        },
        {
            "type": "normal",
            "layer":{
                "layer_pdc": 9,
                "layer_pbw":5.9249,
                "layer_pbh": 8.0710,
                "layer_ppw": 0.547,
                
                # top parameters
                "pattern_tp0": 0.00,
                "pattern_tnn": 2.00,
                "pattern_tmm": 2.00,
                
                # bottom parameters
                "pattern_bp0": 0.00,
                "pattern_bnn": 2.00,
                "pattern_bmm": 2.00,
                
                "pattern_twist": True,
                "pattern_symmetry": True,
            
                "color": "#ffca28",
            }
        },
        {
            "type": "normal",
            "layer":{
                "layer_pdc": 9,
                "layer_pbw":5.9249,
                "layer_pbh": 8.0710,
                "layer_ppw": 0.547,
                
                # top parameters
                "pattern_tp0": 0.00,
                "pattern_tnn": 2.00,
                "pattern_tmm": 2.00,
                
                # bottom parameters
                "pattern_bp0": 0.00,
                "pattern_bnn": 2.00,
                "pattern_bmm": 2.00,
                
                "pattern_twist": True,
                "pattern_symmetry": True,
            
                "color": "#28ff4c",
            }
        },
        {
            "type": "end",
            "layer":{
                "layer_pdc": 9,
                "layer_pbw":5.9249,
                "layer_pbh": 8.0710,
                "layer_ppw": 0.547,
                
                # top parameters
                "pattern_tp0": 0.00,
                "pattern_tnn": 2.00,
                "pattern_tmm": 2.00,
                
                # bottom parameters
                "pattern_bp0": 0.00,
                "pattern_bnn": 2.00,
                "pattern_bmm": 2.00,
                
                "pattern_twist": True,
                "pattern_symmetry": True,
            
                "color": "#5885f8",
            }
        }
    ]
}
