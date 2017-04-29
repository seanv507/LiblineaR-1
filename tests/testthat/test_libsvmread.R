test_libsvmread(){
  expect_equal(dim(x_y$x),c(270,13))
  x_y <- libsvmread('../heart_scale')
  expect_true(abs(mean(x_y$y) - -0.1111111)<1e-6)
  expect_true(abs(sum(x_y$x) - -666.4009)<1e-4)
}