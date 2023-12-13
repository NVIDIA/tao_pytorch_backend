# Usage of new pruning API:

```python
    #################################################
    # Torch Pruning (Begin)
    #################################################
    import torch_pruning as tp
    prune_ratio = 0.2
    granularity = 8
    model.eval()
    num_params_before_pruning = tp.utils.count_params( model )
    # 1. build dependency graph
    DG = tp.DependencyGraph()
    out = model(torch.randn([1,3, input_size, input_size]).to(device))
    DG.build_dependency(model, example_inputs=torch.randn([1,3, input_size, input_size]))
    excluded_layers = list(model.model[-1].modules())
    print(excluded_layers)

    # 2. get global threshold
    global_thresh, module2scores = tp.utils.get_global_thresh(model, prune_ratio=prune_ratio)
    # Hard code the way to find the shortcut connection in YOLOV5 module
    from models.common import C3
    merged_sets = {}
    for name, m in model.named_modules():
        if isinstance(m, C3):
            if m.shortcut:
                merged_sets[m.cv1.conv] = set()
                for btnk in m.m:
                    merged_sets[m.cv1.conv].add(btnk.cv2.conv)

    # 3. Execute pruning
    tp.utils.execute_custom_score_prune(model,
                                        global_thresh=global_thresh,
                                        module2scores=module2scores,
                                        dep_graph=DG,
                                        granularity=granularity,
                                        excluded_layers=excluded_layers,
                                        merged_sets=merged_sets)
    num_params_after_pruning = tp.utils.count_params( model )
    print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))
    # exit(0)
    #################################################
    # Torch Pruning (End)
    #################################################
    model = model.to(device)
```
