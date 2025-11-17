library(magrittr)
library(ggplot2)
library(nnet)

N <- 200 # Point by class
D <- 2 # Dimensionality
K <- 4 # Class numbers
X <- data.frame() # Data matrix
y <- data.frame() # Class Labels

set.seed(308)

for (j in (1:K)) {
  r <- seq(0.05, 1, length.out = N)
  t <- seq((j - 1) * 4.7, j * 4.7, length.out = N) + rnorm(N, sd = 0.3) # theta
  Xtemp <- data.frame(x1 = r * sin(t), x2 = r * cos(t))
  ytemp <- data.frame(matrix(j, N, 1))
  X <- rbind(X, Xtemp)
  y <- rbind(y, ytemp)
}
data <- cbind(X, y)
colnames(data) <- c(colnames(X), 'label')

x1_min <- min(X[, 1]) - 0.2
x1_max <- max(X[, 1]) + 0.2
x2_min <- min(X[, 2]) - 0.2
x2_max <- max(X[, 2]) + 0.2

# Spiral Graph
puntos <-
  ggplot() +
  geom_point(
    data = data,
    aes(x = x1, y = x2, color = as.character(label)),
    size = 2
  ) +
  theme_bw(base_size = 15) +
  xlim(x1_min, x2_max) +
  ylim(x2_min, x2_max) +
  ggtitle('Spiral Data Visualization') +
  coord_fixed(ratio = 1) +
  theme(
    axis.ticks = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    legend.position = 'none'
  )

puntos


hs <- 0.01
grid <- as.matrix(expand.grid(
  x1 = seq(x1_min, x1_max, by = hs),
  x2 = seq(x2_min, x2_max, by = hs)
))

# Use when Neural networks it's ready
#puntos +
#  geom_tile(aes(
#    x = grid[, 1],
#    y = grid[, 2],
#    fill = 'red',
#    alpha = 0.3
#  ))

# One-hot encoding (Y matrix)
# https://r-universe.dev/manuals/nnet.html
# Generates Class Indicator Matrix from a Factor
# class.ind(cl)
# It created de Y matrix, where each 1 in a cell
# represents the respecctive class
Y <- class.ind(data$label)
colnames(Y) <- paste0('C', 1:ncol(Y))

# X matrix, basen in out dataset
# In summary it's just selecting out dataset
# without contemplating label column
Xmat <- as.matrix(data[, c("x1", "x2")])

# Third section, training models
set.seed(123)

# Initialize list to compare results later
models <- list()
results <- list()

# 3a no hidden layers
m0 <- multinom(label ~ x1 + x2, data = data, trace = FALSE, maxit = 1000)
models[["no_hidden_multinom"]] <- m0

# 3b (2, 5, 15) hidden layers
sizes <- c(2, 5, 15)
for (s in sizes) {
  name <- paste0("nnet_size_", s, "_decay_0")
  m <- nnet(
    Xmat,
    Y,
    size = s,
    softmax = TRUE,
    maxit = 1000,
    decay = 0,
    trace = FALSE
  )
  models[[name]] <- m
}

# 3c 15 n in hidden layer and decay paramaters
decays <- c(0.005, 0.05, 0.1, 0.5, 1)
for (d in decays) {
  name <- paste0("nnet_size_15_decay_", gsub("\\.", "_", as.character(d)))
  m <- nnet(
    Xmat,
    Y,
    size = 15,
    softmax = TRUE,
    maxit = 1000,
    decay = d,
    trace = FALSE
  )
  models[[name]] <- m
}

# 4

# Predicción y visualización para cada modelo
for (nm in names(models)) {
  model <- models[[nm]]

  # Calcular predicciones (SOFTMAX: matriz de probabilidades)
  if (inherits(model, "nnet")) {
    # nnet devuelve matriz si softmax = TRUE
    pred_probs <- predict(model, grid)
  } else if (inherits(model, "multinom")) {
    # multinom requiere type = "prob"
    pred_probs <- predict(model, newdata = as.data.frame(grid), type = "prob")
  } else {
    stop("Modelo no reconocido: ", nm)
  }

  # Convertir a matriz si es vector o data.frame
  pred_probs <- as.matrix(pred_probs)

  # Asegurar que tiene dimensiones válidas
  if (is.null(dim(pred_probs))) {
    stop(paste(
      "El modelo",
      nm,
      "no devolvió probabilidades con dimensiones válidas."
    ))
  }

  # Calcular clases (softmax: clase de mayor probabilidad)
  classes <- as.factor(apply(pred_probs, 1, which.max))

  # Graficar (superponer las clases sobre los puntos originales)
  cat("\nMostrando modelo:", nm, "\n")
  print(
    puntos +
      geom_tile(
        aes_string(
          x = "grid[,1]",
          y = "grid[,2]",
          fill = "classes",
          alpha = 0.3
        ),
        show.legend = FALSE
      ) +
      ggtitle(paste("Modelo:", nm))
  )
}
