@echo off

for /l %%r in (0, 1, 2) do (
  for /l %%c in (0, 1, 9) do (
      call Java -jar target/MoFGBML-23.0.0-SNAPSHOT.jar iris FAN2021 trial%%r%%c 1 dataset\iris\a%%r_%%c_iris-10tra.dat dataset\iris\a%%r_%%c_iris-10tst.dat
  )
)