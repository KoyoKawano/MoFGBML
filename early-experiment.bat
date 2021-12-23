@echo off

for /l %%r in (0, 1, 2) do (
  for /l %%c in (0, 1, 9) do (
      call Java -jar target/MoFGBML-23.0.0-SNAPSHOT.jar pima pima trial%%r%%c 1 dataset\pima\a%%r_%%c_pima-10tra.dat dataset\pima\a%%r_%%c_pima-10tst.dat 1
  )
)
