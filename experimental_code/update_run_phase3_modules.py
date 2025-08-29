        # Module 3: Attractor Inversion
        if 3 in modules:
            attractor_inverter = run_phase3_module3(args, logger, boundary_mapper)
        else:
            attractor_inverter = None
        
        # Module 4: Morphospace Projection
        if 4 in modules:
            morphospace_projector = run_phase3_module4(args, logger, 
                                                     [boundary_mapper, nudging_controller, attractor_inverter])