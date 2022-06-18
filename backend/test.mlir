#loc0 = loc(unknown)
module @jit_prim_fun.12 {
  func.func public @main(%arg0: tensor<1x1xi32> loc(unknown), %arg1: tensor<1xi32> loc(unknown), %arg2: tensor<1xi32> loc(unknown)) -> tensor<1x1xi32> {
    %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<i32> loc(unknown), %arg4: tensor<i32> loc(unknown)):
      "mhlo.return"(%arg4) : (tensor<i32>) -> () loc(#loc1)
    }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1xi32> loc(#loc1)
    return %0 : tensor<1x1xi32> loc(#loc0)
  } loc(#loc0)
  } loc(#loc0)
#loc1 = loc("jit(scatter)/jit(main)/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(0,), inserted_window_dims=(1,), scatter_dims_to_operand_dims=(1,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"("/Users/birch/anaconda3/envs/torch-nightly/lib/python3.9/site-packages/transformers/models/bart/modeling_flax_bart.py":926:1))