ASMDIR=/tmp/IntelIGC/cc_test
OUTDIR=/home/adstraw/work/mrpt
mkdir $OUTDIR

echo "Max Regs, Actual Regs, Accesses, Cache Misses, Best Plan, Mem Bounded?, Groups, Mem IO, TotOps, CB Goodness, MB Cost, Kernel Time, ASM, Spill Size, GRF Ref Count, BAD" > $OUTDIR/mrpt.csv

i=1
while [ $i -le 980 ]; do
  rm -rf $ASMDIR/*
  PLAIDML_COUNTER=$i MAX_REGS_PER_THREAD=128 IGC_ShaderDumpEnable=1 PLAIDML_DUMP=$ASMDIR bazelisk run //plaidml/edsl/tests:cc_test -- --plaidml_device=opencl.0 --plaidml_target="intel_gen_ocl_spirv{spirv-version=120}" --gtest_filter=*Res2a2a*

  # LAYER 2A, BRANCH 2A KERNELS
  rm -rf $ASMDIR/OCL_asm696f72efc982cec6_simd32_f.asm

  # TODO there could be more than one, still
  ASM=$(ls $ASMDIR/*.asm)
  echo -n $ASM >> $OUTDIR/mrpt.csv
  echo -n ", " >> $OUTDIR/mrpt.csv

  SPILLSIZE=$(cat $ASM | grep "\.spill size" | sed 's|\/\/.spill size \([0-9]*\)|\1|g')
  echo -n $SPILLSIZE >> $OUTDIR/mrpt.csv
  echo -n ", " >> $OUTDIR/mrpt.csv

  GRFCOUNT=$(cat $ASM | grep "\.spill GRF" | sed 's|\/\/.spill GRF ref count \([0-9]*\)|\1|g')
  echo -n $GRFCOUNT >> $OUTDIR/mrpt.csv
  echo -n ", " >> $OUTDIR/mrpt.csv

  BAD=$(cat $ASM | grep "BAD:" | sed 's|\/\/ --  BAD: \([0-9]*\)|\1|g')
  echo -n $BAD >> $OUTDIR/mrpt.csv
  echo "" >> $OUTDIR/mrpt.csv

  mv $ASM $OUTDIR

  i=$((i + 1))
done
