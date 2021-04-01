// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine   \
// RUN:     -loop-invariant-code-motion -lower-affine   \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e xsmm | FileCheck %s

!I_memref = type memref<1x16x16x16xf32>
!K_memref = type memref<3x3x16x16xf32>
!O_memref = type memref<1x14x14x16xf32>

func private @print_memref_f32(memref<*xf32>)


// Reference implementation.

#input_map = affine_map<(d0, d1)[u] -> (u * d0 + d1)>


func @gold_conv_forward(%A: !I_memref, %B: !K_memref, %C: !O_memref){
    %u = constant 1 : index
    %v = constant 1 : index
    %zero = constant 0 : index
    %step = constant 1: index
 
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %N = dim %C, %c0 : !O_memref
    %P = dim %C, %c1 : !O_memref
    %Q = dim %C, %c2 : !O_memref
    %K = dim %C, %c3 : !O_memref
    %C1 = dim %A, %c3 : !I_memref
    %R = dim %B, %c0 : !K_memref
    %S = dim %B, %c1: !K_memref


    scf.for %n = %zero to %N step %step {
      scf.for %k = %zero to %K step %step {
        scf.for %c = %zero to %C1 step %step {
          scf.for %p = %zero to %P step %step {
            scf.for %q = %zero to %Q step %step {
              scf.for %r = %zero to %R step %step {
                scf.for %s = %zero to %S step %step {
                  %upr = affine.apply #input_map(%p, %r)[%u]
                  %vqs = affine.apply #input_map(%q, %s)[%v]
                  %ld.a = load %A[%n, %upr, %vqs, %c] : !I_memref
                  %ld.b = load %B[%r, %s, %c, %k] : !K_memref
                  %ld.c = load %C[%n, %p, %q, %k] : !O_memref
                  %mul.ab = mulf %ld.a, %ld.b : f32
                  %sum.ab.c = addf %ld.c, %mul.ab : f32
                  store %sum.ab.c, %C[%n, %p, %q, %k] : !O_memref
                }
              }
            }
          }
        }
      }
    }
    return
}




func @check_output(%A : memref<?x?x?x?xf32>, %B : memref<?x?x?x?xf32>, %C: memref<1xf32>)->(){
  %ind = constant 0 : index
  %res = alloca(): memref<1xi32>
  %zero = constant 0: index
  %one = constant 1: i32
  %step = constant 1: index
  %lb = constant 0:index

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %N = dim %A, %c0 : memref<?x?x?x?xf32>
  %P = dim %A, %c1 : memref<?x?x?x?xf32>
  %Q = dim %A, %c2 : memref<?x?x?x?xf32>
  %K = dim %A, %c3 : memref<?x?x?x?xf32>





  store %one, %res[%ind]: memref<1 x i32>
 
  scf.for %i0 = %lb to %N step %step {
    scf.for %j0 =%lb to %P step %step {
      scf.for %r0 = %lb to %Q step %step {
        scf.for %s0 = %lb to %K step %step {
          %res2 = load %res[%ind] : memref<1 x i32>
          %a = load %A[%i0, %j0, %r0, %s0] :memref<?x?x?x?xf32>
          %b = load %B[%i0, %j0, %r0, %s0] :memref<?x?x?x?xf32>
          %p = cmpf "oeq", %a , %b : f32
          %dummy_one = trunci %one: i32 to i1
          %interm = and %p, %dummy_one: i1
          %dummy = trunci %res2: i32 to i1
          %res3 = and %dummy, %interm: i1
          %res4 = zexti %res3:i1 to i32
          store %res4, %res[%ind]: memref<1xi32>
        }
      }
    }
  }
 
  %interm = load %res[%ind] : memref<1 x i32>
  %succ = sitofp %interm : i32 to f32
  store %succ, %C[%c0]: memref<1xf32>
  return
}


func @xsmm() {

  %ref_conv = constant @pad_contraction : (!I_memref, !K_memref, !O_memref) -> ()
  %conv2 = constant @gold_conv_forward : (!I_memref, !K_memref, !O_memref) -> ()
  call @test_conv2(%ref_conv, %conv2) : ((!I_memref, !K_memref, !O_memref) -> (), (!I_memref, !K_memref, !O_memref) -> ()) -> ()

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






func @pad_contraction(%arg0: memref<1x16x16x16xf32>, %arg1: memref<3x3x16x16xf32>, %arg2: memref<1x14x14x16xf32>) {
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 14 step 14 {
        affine.for %arg5 = 0 to 14 {
          affine.for %arg6 = 0 to 16 step 16 {
            affine.for %arg7 = 0 to 3 step 3 {
              affine.for %arg8 = 0 to 3 step 3 {
                affine.for %arg9 = 0 to 16 step 16 {
                  %0 = affine.load %arg0[%arg3, %arg7 + %arg4, %arg8 + %arg5, %arg9] : memref<1x16x16x16xf32>
                  %1 = affine.load %arg1[%arg7, %arg8, %arg9, %arg6] : memref<3x3x16x16xf32>
                  %2 = mulf %0, %1 : f32
                  %3 = addi %arg7, %arg4 : index
                  %4 = addi %arg8, %arg5 : index
                  %5 = xsmm.brgemm.offs.dispatch.f32 [14, 16, 16], [256, 16, 224]
                  "xsmm.brgemm.offs.invoke.f32"(%5, %arg2, %arg0, %arg1, %arg3, %arg4, %arg5, %arg6, %arg3, %3, %4, %arg9, %arg7, %arg8, %arg9, %arg6) {aOffsets = [0, 1024, 2048, 64, 1088, 2112, 128, 1152, 2176], bOffsets = [0, 3072, 6144, 1024, 4096, 7168, 2048, 5120, 8192], numBatches = 9 : i64} : (i64, memref<1x14x14x16xf32>, memref<1x16x16x16xf32>, memref<3x3x16x16xf32>, index, index, index, index, index, index, index, index, index, index, index, index) -> ()
                }
              }
            }
          }
        }
      }
    }
    return
  }





func @test_conv2(%impl : (!I_memref, !K_memref, !O_memref) -> (), %impl2 : (!I_memref, !K_memref, !O_memref) -> ()) {
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

  %O_2 = alloc() : !O_memref
  %O_2_2d = memref_cast %O_2 : !O_memref to memref<?x?x?x?xf32>
  %O_2_ud = memref_cast %O_2 : !O_memref to memref<*xf32>
  linalg.fill(%O_2, %f0) : !O_memref, f32




  call_indirect %impl(%I, %K, %O) : (!I_memref, !K_memref, !O_memref) -> ()
  call_indirect %impl2(%I, %K, %O_2) : (!I_memref, !K_memref, !O_memref) -> ()

  //call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()

  //call @print_memref_f32(%O_2_ud) : (memref<*xf32>) -> ()
  
  %output_check = alloc() : memref<1xf32>
  call @check_output(%O_2d, %O_2_2d, %output_check) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<1xf32>) -> ()
  %output_check_cast = memref_cast %output_check : memref<1xf32> to memref<*xf32>
  call @print_memref_f32(%output_check_cast) : (memref<*xf32>) -> ()
  //CHECK: [1]
  dealloc %O_2 : !O_memref
  dealloc %O : !O_memref
  dealloc %K : !K_memref
  dealloc %I : !I_memref
  return
}
