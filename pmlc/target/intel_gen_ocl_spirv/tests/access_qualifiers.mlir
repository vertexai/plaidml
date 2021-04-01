// RUN: pmlc-opt --intel-gen-ocl-set-access-qualifiers %s | FileCheck %s

spv.module @access_qualifier1 Physical64 OpenCL {
  spv.func @kernel(%arg0: !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0])>, CrossWorkgroup>) "None" attributes {spv.entry_point_abi = {local_size = dense<[16, 3, 3]> : vector<3xi32>}, workgroup_attributions = 0 : i64} {
	// CHECK: spv.AccessChain {{.*}}[{{.*}}, {{.*}}] : !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0, FuncParamAttr=6])>
	%0 = spv.Constant 0 : i32
    %1 = spv.AccessChain %arg0[%0, %0] : !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0])>, CrossWorkgroup>, i32, i32
	%2 = spv.Load "CrossWorkgroup" %1 : f32
    spv.Return
  }
}

spv.module @access_qualifier2 Physical64 OpenCL {
  spv.func @kernel(%arg0: !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0])>, CrossWorkgroup>) "None" attributes {spv.entry_point_abi = {local_size = dense<[16, 3, 3]> : vector<3xi32>}, workgroup_attributions = 0 : i64} {
	// CHECK: spv.AccessChain {{.*}}[{{.*}}, {{.*}}] : !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0, FuncParamAttr=6])>
	%0 = spv.Constant 0 : i32
    %1 = spv.AccessChain %arg0[%0, %0] : !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0])>, CrossWorkgroup>, i32, i32
	%2 = spv.SubgroupBlockReadINTEL "CrossWorkgroup" %1 : f32
    spv.Return
  }
}

spv.module @access_qualifier3 Physical64 OpenCL {
  spv.func @kernel(%arg0: !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0])>, CrossWorkgroup>) "None" attributes {spv.entry_point_abi = {local_size = dense<[16, 3, 3]> : vector<3xi32>}, workgroup_attributions = 0 : i64} {
	// CHECK: spv.Bitcast {{.*}} : !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0, FuncParamAttr=6])>, CrossWorkgroup>
	%0 = spv.Constant 0 : i32
	%1 = spv.Bitcast %arg0 : !spv.ptr<!spv.struct<(!spv.array<64 x f32, stride=4> [0])>, CrossWorkgroup> to !spv.ptr<!spv.struct<(!spv.array<64 x i32, stride=4> [0])>, CrossWorkgroup>
    %2 = spv.AccessChain %1[%0, %0] : !spv.ptr<!spv.struct<(!spv.array<64 x i32, stride=4> [0])>, CrossWorkgroup>, i32, i32
	%3 = spv.Load "CrossWorkgroup" %2 : i32
    spv.Return
  }
}
