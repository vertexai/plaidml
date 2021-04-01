// RUN: pmlc-opt --intel-gen-ocl-legalize-spirv %s | FileCheck %s

spv.module @__spv__main_kernel Physical64 OpenCL requires #spv.vce<v1.0, [Kernel, Addresses], []> {
  // CHECK: spv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi64>, Input>
  // CHECK: spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi64>, Input>
  // CHECK: spv.GlobalVariable @__builtin_var_SubgroupId__ built_in("SubgroupId") : !spv.ptr<i64, Input>
  spv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
  spv.GlobalVariable @__builtin_var_SubgroupId__ built_in("SubgroupId") : !spv.ptr<i32, Input>
  spv.func @main_kernel() "None" attributes {workgroup_attributions = 0 : i64} {
    %2 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %3 = spv.Load "Input" %2 : vector<3xi32>
    // CHECK: spv.CompositeExtract {{.*}}: vector<3xi32>
    %4 = spv.CompositeExtract %3[0 : i32] : vector<3xi32>
    %6 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %7 = spv.Load "Input" %6 : vector<3xi32>
    // CHECK: spv.CompositeExtract {{.*}}: vector<3xi32>
    %8 = spv.CompositeExtract %7[0 : i32] : vector<3xi32>
    %9 = spv.mlir.addressof @__builtin_var_SubgroupId__ : !spv.ptr<i32, Input>
    // CHECK: spv.UConvert {{.*}}: i64 to i32
    %10 = spv.Load "Input" %9 : i32
    spv.Return
  }
  spv.EntryPoint "Kernel" @main_kernel, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__, @__builtin_var_SubgroupId__
}
