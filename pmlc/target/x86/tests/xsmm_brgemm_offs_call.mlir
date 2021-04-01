// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e baseline | FileCheck %s
// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e tiled | FileCheck %s
// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine \
// RUN:     -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e xsmm | FileCheck %s
// RUN: pmlc-opt -convert-linalg-to-loops \
// RUN:     --pass-pipeline='func(x86-affine-stencil-xsmm{batched=true})' \
// RUN:     -x86-convert-pxa-to-affine \
// RUN:     -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e baseline | FileCheck %s

!I_memref = type memref<1x6x5x7xf32>
!K_memref = type memref<1x1x7x11xf32>
!O_memref = type memref<1x6x5x11xf32>

func private @print_memref_f32(memref<*xf32>)

func @baseline() {
  %dot = constant @dot : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @test_dot(%dot) : ((memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()) -> ()

  %conv2 = constant @conv2 : (!I_memref, !K_memref, !O_memref) -> ()
  call @test_conv2(%conv2) : ((!I_memref, !K_memref, !O_memref) -> ()) -> ()

  return
}

func @tiled() {
  %dot = constant @dot_tiled : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @test_dot(%dot) : ((memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()) -> ()

  %conv2 = constant @conv2_tiled : (!I_memref, !K_memref, !O_memref) -> ()
  call @test_conv2(%conv2) : ((!I_memref, !K_memref, !O_memref) -> ()) -> ()

  return
}

func @xsmm() {
  %dot = constant @dot : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @test_dot(%dot) : ((memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()) -> ()

  %conv2 = constant @conv2_xsmm : (!I_memref, !K_memref, !O_memref) -> ()
  call @test_conv2(%conv2) : ((!I_memref, !K_memref, !O_memref) -> ()) -> ()

  return
}

func @fill_2d(%buf : memref<?x?xf32>, %alt : i1) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c5 = constant 5 : index
  %X = dim %buf, %c0 : memref<?x?xf32>
  %Y = dim %buf, %c1 : memref<?x?xf32>
  affine.parallel (%x, %y) = (0, 0) to (%X, %Y) {
    // i = linear offset
    %i = affine.apply affine_map<(x, y)[Y] -> (x * Y + y)>(%x, %y)[%Y]
    // t = alt ? i : 0
    %t = select %alt, %i, %c0 : index
    // v = x + y + t - 5
    %1 = addi %x, %y : index
    %2 = addi %1, %t : index
    %v = subi %2, %c5 : index
    %v_i64 = index_cast %v : index to i64
    %v_f32 = sitofp %v_i64 : i64 to f32
    store %v_f32, %buf[%x, %y] : memref<?x?xf32>
  }
  return
}

func @fill_4d(%buf : memref<?x?x?x?xf32>, %alt : i1) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c5 = constant 5 : index
  %X = dim %buf, %c0 : memref<?x?x?x?xf32>
  %Y = dim %buf, %c1 : memref<?x?x?x?xf32>
  %Z = dim %buf, %c2 : memref<?x?x?x?xf32>
  %W = dim %buf, %c3 : memref<?x?x?x?xf32>
  affine.parallel (%x, %y, %z, %w) = (0, 0, 0, 0) to (%X, %Y, %Z, %W) {
    // i = linear offset
    %i = affine.apply affine_map<(x, y, z, w)[Y, Z, W] -> (x * Y + y * W + z * W + w)>(%x, %y, %z, %w)[%Y, %Z, %W]
    // t = alt ? i : 0
    %t = select %alt, %i, %c0 : index
    %j = affine.apply affine_map<(x, y, z, w) -> (x + y + z + w)>(%x, %y, %z, %w)
    // v = j + t - 5
    %2 = addi %j, %t : index
    %v = subi %2, %c5 : index
    %v_i64 = index_cast %v : index to i64
    %v_f32 = sitofp %v_i64 : i64 to f32
    store %v_f32, %buf[%x, %y, %z, %w] : memref<?x?x?x?xf32>
  }
  return
}

func @test_dot(%impl : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()) {
  %false = constant 0 : i1
  %true = constant 1 : i1
  %f0 = constant 0.0 : f32
  %A = alloc() : memref<8x8xf32>
  %A_2d = memref_cast %A : memref<8x8xf32> to memref<?x?xf32>
  %A_ud = memref_cast %A : memref<8x8xf32> to memref<*xf32>
  call @fill_2d(%A_2d, %false) : (memref<?x?xf32>, i1) -> ()
  %B = alloc() : memref<8x8xf32>
  %B_2d = memref_cast %B : memref<8x8xf32> to memref<?x?xf32>
  %B_ud = memref_cast %B : memref<8x8xf32> to memref<*xf32>
  call @fill_2d(%B_2d, %true) : (memref<?x?xf32>, i1) -> ()
  %C = alloc() : memref<8x8xf32>
  %C_2d = memref_cast %C : memref<8x8xf32> to memref<?x?xf32>
  %C_ud = memref_cast %C : memref<8x8xf32> to memref<*xf32>

  linalg.fill(%C, %f0) : memref<8x8xf32>, f32
  call_indirect %impl(%A_2d, %B_2d, %C_2d) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @print_memref_f32(%C_ud) : (memref<*xf32>) -> ()
  // CHECK:  [60,   36,   12,   -12,   -36,   -60,   -84,   -108],
  // CHECK:  [272,   264,   256,   248,   240,   232,   224,   216],
  // CHECK:  [484,   492,   500,   508,   516,   524,   532,   540],
  // CHECK:  [696,   720,   744,   768,   792,   816,   840,   864],
  // CHECK:  [908,   948,   988,   1028,   1068,   1108,   1148,   1188],
  // CHECK:  [1120,   1176,   1232,   1288,   1344,   1400,   1456,   1512],
  // CHECK:  [1332,   1404,   1476,   1548,   1620,   1692,   1764,   1836],
  // CHECK:  [1544,   1632,   1720,   1808,   1896,   1984,   2072,   2160]

  dealloc %C : memref<8x8xf32>
  dealloc %B : memref<8x8xf32>
  dealloc %A : memref<8x8xf32>
  return
}

func @dot(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = dim %C, %c0 : memref<?x?xf32>
  %N = dim %C, %c1 : memref<?x?xf32>
  %K = dim %A, %c1 : memref<?x?xf32>
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (%M, %N, %K) {
    %0 = affine.load %A[%i, %k] : memref<?x?xf32>
    %1 = affine.load %B[%k, %j] : memref<?x?xf32>
    %2 = mulf %0, %1 : f32
    pxa.reduce addf %2, %C[%i, %j] : memref<?x?xf32>
  }
  return
}

func @dot_tiled(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = constant 1 : index
  %c1 = constant 1 : index
  %M = dim %C, %c0 : memref<?x?xf32>
  %N = dim %C, %c1 : memref<?x?xf32>
  %K = dim %A, %c1 : memref<?x?xf32>
  affine.parallel (%i0, %j0, %k0) = (0, 0, 0) to (%M, %N, %K) step (2, 2, 2) {
    affine.parallel (%i1, %j1, %k1) = (%i0, %j0, %k0) to (%i0 + 2, %j0 + 2, %k0 + 2) {
      %0 = affine.load %A[%i1, %k1] : memref<?x?xf32>
      %1 = affine.load %B[%k1, %j1] : memref<?x?xf32>
      %2 = mulf %0, %1 : f32
      pxa.reduce addf %2, %C[%i1, %j1] : memref<?x?xf32>
    }
  }
  return
}

func @test_conv2(%impl : (!I_memref, !K_memref, !O_memref) -> ()) {
  %false = constant 0 : i1
  %true = constant 1 : i1
  %f0 = constant 0.0 : f32
  %I = alloc() : !I_memref
  %I_2d = memref_cast %I : !I_memref to memref<?x?x?x?xf32>
  %I_ud = memref_cast %I : !I_memref to memref<*xf32>
  call @fill_4d(%I_2d, %false) : (memref<?x?x?x?xf32>, i1) -> ()
  %K = alloc() : !K_memref
  %K_2d = memref_cast %K : !K_memref to memref<?x?x?x?xf32>
  %K_ud = memref_cast %K : !K_memref to memref<*xf32>
  call @fill_4d(%K_2d, %true) : (memref<?x?x?x?xf32>, i1) -> ()
  %O = alloc() : !O_memref
  %O_2d = memref_cast %O : !O_memref to memref<?x?x?x?xf32>
  %O_ud = memref_cast %O : !O_memref to memref<*xf32>

  linalg.fill(%O, %f0) : !O_memref, f32
  call_indirect %impl(%I, %K, %O) : (!I_memref, !K_memref, !O_memref) -> ()
  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()
  // CHECK: [-98,     -126,     -154,     -182,     -210,     -238,     -266,     -294,     -322,     -350,     -378],
  // CHECK: [119,     105,     91,     77,     63,     49,     35,     21,     7,     -7,     -21],
  // CHECK: [336,     336,     336,     336,     336,     336,     336,     336,     336,     336,     336],
  // CHECK: [553,     567,     581,     595,     609,     623,     637,     651,     665,     679,     693],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050]],
  // CHECK: [119,     105,     91,     77,     63,     49,     35,     21,     7,     -7,     -21],
  // CHECK: [336,     336,     336,     336,     336,     336,     336,     336,     336,     336,     336],
  // CHECK: [553,     567,     581,     595,     609,     623,     637,     651,     665,     679,     693],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407]],
  // CHECK: [336,     336,     336,     336,     336,     336,     336,     336,     336,     336,     336],
  // CHECK: [553,     567,     581,     595,     609,     623,     637,     651,     665,     679,     693],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407],
  // CHECK: [1204,     1260,     1316,     1372,     1428,     1484,     1540,     1596,     1652,     1708,     1764]],
  // CHECK: [553,     567,     581,     595,     609,     623,     637,     651,     665,     679,     693],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407],
  // CHECK: [1204,     1260,     1316,     1372,     1428,     1484,     1540,     1596,     1652,     1708,     1764],
  // CHECK: [1421,     1491,     1561,     1631,     1701,     1771,     1841,     1911,     1981,     2051,     2121]],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407],
  // CHECK: [1204,     1260,     1316,     1372,     1428,     1484,     1540,     1596,     1652,     1708,     1764],
  // CHECK: [1421,     1491,     1561,     1631,     1701,     1771,     1841,     1911,     1981,     2051,     2121],
  // CHECK: [1638,     1722,     1806,     1890,     1974,     2058,     2142,     2226,     2310,     2394,     2478]],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407],
  // CHECK: [1204,     1260,     1316,     1372,     1428,     1484,     1540,     1596,     1652,     1708,     1764],
  // CHECK: [1421,     1491,     1561,     1631,     1701,     1771,     1841,     1911,     1981,     2051,     2121],
  // CHECK: [1638,     1722,     1806,     1890,     1974,     2058,     2142,     2226,     2310,     2394,     2478],
  // CHECK: [1855,     1953,     2051,     2149,     2247,     2345,     2443,     2541,     2639,     2737,     2835]]]]

  dealloc %O : !O_memref
  dealloc %K : !K_memref
  dealloc %I : !I_memref
  return
}

func @conv2(%I: !I_memref, %K: !K_memref, %O: !O_memref) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c1 : !I_memref
  %Y = dim %I, %c2 : !I_memref
  %CI = dim %I, %c3 : !I_memref
  %CO = dim %O, %c3 : !O_memref
  affine.parallel (%x, %y, %ci, %co) = (0, 0, 0, 0) to (%X, %Y, %CI, %CO) {
    %0 = affine.load %I[0, %x, %y, %ci] : !I_memref
    %1 = affine.load %K[0, 0, %ci, %co] : !K_memref
    %2 = mulf %0, %1 : f32
    pxa.reduce addf %2, %O[0, %x, %y, %co] : !O_memref
  }
  return
}

func @conv2_tiled(%I: !I_memref, %K: !K_memref, %O: !O_memref) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c1 : !I_memref
  %Y = dim %I, %c2 : !I_memref
  %CI = dim %I, %c3 : !I_memref
  %CO = dim %O, %c3 : !O_memref
  affine.parallel (%x0, %y) = (0, 0) to (%X, %Y) step (2, 1) {
    affine.parallel (%x1, %ci, %co) = (%x0, 0, 0) to (%x0 + 2, %CI, %CO) {
      %0 = affine.load %I[0, %x1, %y, %ci] : !I_memref
      %1 = affine.load %K[0, 0, %ci, %co] : !K_memref
      %2 = mulf %0, %1 : f32
      pxa.reduce addf %2, %O[0, %x1, %y, %co] : !O_memref
    }
  }
  return
}


#O_tile = affine_map<(m, n) -> (0, m, 0, n)>
#I_tile = affine_map<(m, k) -> (0, m, 0, k)>
#K_tile = affine_map<(k, n) -> (0, 0, k, n)>



func @conv2_xsmm(%I: !I_memref, %K: !K_memref, %O: !O_memref) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c1 : !I_memref
  %Y = dim %I, %c2 : !I_memref
  %CI = dim %I, %c3 : !I_memref
  %CO = dim %O, %c3 : !O_memref
  %ptr = xsmm.brgemm.offs.dispatch.f32 [2, 11, 1], [35, 11, 55]
  affine.parallel (%x, %y) = (0, 0) to (%X, %Y) step (2, 1) {
    xsmm.brgemm.offs.invoke.f32 %ptr, %O[%c0, %x, %y, %c0] = %I[%c0, %x, %y, %c0], %K[%c0, %c0, %c0, %c0], 7, [0,4,8,12,16,20,24], [0,44,88,132,176,220,264] : (!I_memref, !K_memref) -> !O_memref
  }
  return


}
