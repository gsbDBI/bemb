install.packages("mlogit")
library("mlogit")
data("ModeCanada", package = "mlogit")
MC <- dfidx(ModeCanada, subset = noalt == 4)
ml.MC1 <- mlogit(choice ~ cost + freq + ovt | income | ivt, MC, reflevel='air')

summary(ml.MC1)