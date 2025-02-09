# ______________________________________________________________________________
#
# Logistische Elastic-Net Regression (E-Net)
#
# ______________________________________________________________________________

res1_b1 <- NULL # Ergbnisobjekte der Prädiktorenblöcke außerhalb des Loops erstellen
res1_b2 <- NULL # b1 = Block; b2 = Block 2; b3 = Block 3, b1_b2_b3 = alle Blöcke
res1_b3 <- NULL
res1_b1_b2_b3 <- NULL



for (i in 1:100) { # Outer Loop mit einer Iteration, um alle E-Net Modelle zu schätzen
  print(paste0(i, ". Iter"))
  
  set.seed(20190930 + i) # Seed gewährleistet Reproduzierbarkeit der Ergebnisse
  trainIndex <- createDataPartition(data_final$replicate, # zufällige Partitionierung des Datensatzes in Trainings- und Testdatensatz
                                    p = .8, # 80% Trainingsdaten; 20% Testdaten
                                    list = FALSE,
                                    times = 1)
  train_dat <- data_final[ trainIndex, ] # Speichern der beiden Partitionen
  test_dat  <- data_final[-trainIndex, ] 
  
  # Pre-Processing: Transformation der Daten nach Partitionierung um Data-Leakage zu vermeiden
  preProcValues_train <- preProcess(train_dat, method = c("knnImpute")) # Imputieren fehlender Daten + center + scale 
  train_dat <- predict(preProcValues_train, train_dat) # speichern der z-Standardisierten Trainingsdaten
  
  preProcValues_test <- preProcess(test_dat, method = c("knnImpute")) # Imputieren fehlender Daten + center + scale 
  test_dat <- predict(preProcValues_test, test_dat) # speichern der z-Standardisierten Testsdaten
  
  # grid <- expand.grid(alpha = seq(0, 1, length = 11),
                      # lambda = seq(0.001, 0.1, length = 21)) # Art des Hyperparametertunings
  

  #### Modelle ####
  
  #### Analyse Prädiktoren Block 1 ####
  
  # Training des Elastic-Net-Modells für Block 1
  enet1_b1 <- train(as.formula(mod_b1), # zuvor spezifiziertes Modell
                   train_dat[, c(Präd_b1, "replicate")], # Trainingsdaten: Prädiktoren aus Block 1 + "replicate"
                   metric = "logLoss", # logLoss als Metrik zur Bewertung von Klassifikationsfragestellungen
                   method = "glmnet",  # GLM (generalisiertes lineares Modell) zur Schätzung und Regularisierung
                   family = "binomial", # Binäre Klassifikation

                   # Hyperparametertuning
                   trControl = trainControl(method = "LOOCV", # LOOCV
                                            summaryFunction = mnLogLoss, # mean der logLoss
                                            classProbs = TRUE, # Klassenwahrscheinlichkeiten
                                            savePredictions = TRUE), # Vorhersagen speichern

                   # tuneGrid = grid)
                   tuneLength = 21)

  # Das anhand der Trainingsdaten trainierte Modell zur Vorhersagend des Testdatensatzes verwenden
  pred_train_b1 <- predict(enet1_b1, train_dat) # Trainiertes Modell zur Vorhersage der Trainingsdaten verwenden
  pred_test_b1 <- predict(enet1_b1, test_dat) # Trainiertes Modell zur Vorhersage der Testsdaten verwenden

  cm_train_b1 <- confusionMatrix(pred_train_b1, train_dat$replicate, positive = "yes") # Konfusionsmatrix Trainingsdaten
  cm_test_b1 <- confusionMatrix(pred_test_b1, test_dat$replicate, positive = "yes") # Konfusionsmatrix Testdaten

  # Speichern der Ergebnisse jeder Iteration für Block 1
  res1_b1 <- rbind(res1_b1, cbind('iter' = i,
                                'mod_b1' = 1:2,
                                'Balanced Accuracy' = c(mean(cm_train_b1$byClass['Balanced Accuracy']), mean(cm_test_b1$byClass['Balanced Accuracy'])), # Als Output-Variablen
                                'Sensitivity' = c(mean(cm_train_b1$byClass['Sensitivity']), mean(cm_test_b1$byClass['Sensitivity'])),                   # BACC, Sen und Spe
                                'Specificity' = c(mean(cm_train_b1$byClass['Specificity']), mean(cm_test_b1$byClass['Specificity']))))                  # definiert.

  #### Analyse für Block 2 analog zu der in Block 2 ####
  
  # Training des Elastic-Net-Modells für Block 2
enet1_b2 <- train(as.formula(mod_b2), # zuvor spezifiziertes Modell
                  train_dat[, c(Präd_b2, "replicate")], # Trainingsdaten: Prädiktoren aus Block 1 + "replicate"
                  metric = "logLoss", # logLoss als Metrik zur Bewertung von Klassifikationsfragestellungen
                  method = "glmnet",  # GLM (generalisiertes lineares Modell) zur Schätzung und Regularisierung
                  family = "binomial", # Binäre Klassifikation

                  # Hyperparametertuning
                  trControl = trainControl(method = "LOOCV", # LOOCV
                                           summaryFunction = mnLogLoss, # mean der logLoss
                                           classProbs = TRUE, # Klassenwahrscheinlichkeiten
                                           savePredictions = TRUE), # Vorhersagen speichern

                  # tuneGrid = grid)
                  tuneLength = 21)

# Das anhand der Trainingsdaten trainierte Modell zur Vorhersagend des Testdatensatzes verwenden
pred_train_b2 <- predict(enet1_b2, train_dat) # Trainiertes Modell zur Vorhersage der Trainingsdaten verwenden
pred_test_b2 <- predict(enet1_b2, test_dat) # Trainiertes Modell zur Vorhersage der Testsdaten verwenden

cm_train_b2 <- confusionMatrix(pred_train_b2, train_dat$replicate, positive = "yes") # Konfusionsmatrix Trainingsdaten
cm_test_b2 <- confusionMatrix(pred_test_b2, test_dat$replicate, positive = "yes") # Konfusionsmatrix Testdaten

# Speichern der Ergebnisse jeder Iteration für Block 2
res1_b2 <- rbind(res1_b2, cbind('iter' = i,
                                'mod_b2' = 1:2,
                                'Balanced Accuracy' = c(mean(cm_train_b2$byClass['Balanced Accuracy']), mean(cm_test_b2$byClass['Balanced Accuracy'])), # Als Output-Variablen
                                'Sensitivity' = c(mean(cm_train_b2$byClass['Sensitivity']), mean(cm_test_b2$byClass['Sensitivity'])),                   # BACC, Sen und Spe
                                'Specificity' = c(mean(cm_train_b2$byClass['Specificity']), mean(cm_test_b2$byClass['Specificity']))))                  # definiert.

  #### Analyse für Block 3 analog zu der in Block 1 ####
  
# Training des Elastic-Net-Modells für Block 3
enet1_b3 <- train(as.formula(mod_b3), # zuvor spezifiziertes Modell
                  train_dat[, c(Präd_b3, "replicate")], # Trainingsdaten: Prädiktoren aus Block 1 + "replicate"
                  metric = "logLoss", # logLoss als Metrik zur Bewertung von Klassifikationsfragestellungen
                  method = "glmnet",  # GLM (generalisiertes lineares Modell) zur Schätzung und Regularisierung
                  family = "binomial", # Binäre Klassifikation

                  # Hyperparametertuning
                  trControl = trainControl(method = "LOOCV", # LOOCV
                                           summaryFunction = mnLogLoss, # mean der logLoss
                                           classProbs = TRUE, # Klassenwahrscheinlichkeiten
                                           savePredictions = TRUE), # Vorhersagen speichern

                  # tuneGrid = grid)
                  tuneLength = 21)

# Das anhand der Trainingsdaten trainierte Modell zur Vorhersagend des Testdatensatzes verwenden
pred_train_b3 <- predict(enet1_b3, train_dat) # Trainiertes Modell zur Vorhersage der Trainingsdaten verwenden
pred_test_b3 <- predict(enet1_b3, test_dat) # Trainiertes Modell zur Vorhersage der Testsdaten verwenden

cm_train_b3 <- confusionMatrix(pred_train_b3, train_dat$replicate, positive = "yes") # Konfusionsmatrix Trainingsdaten
cm_test_b3 <- confusionMatrix(pred_test_b3, test_dat$replicate, positive = "yes") # Konfusionsmatrix Testdaten

# Speichern der Ergebnisse jeder Iteration für Block 3
res1_b3 <- rbind(res1_b3, cbind('iter' = i,
                                'mod_b3' = 1:2,
                                'Balanced Accuracy' = c(mean(cm_train_b3$byClass['Balanced Accuracy']), mean(cm_test_b3$byClass['Balanced Accuracy'])), # Als Output-Variablen
                                'Sensitivity' = c(mean(cm_train_b3$byClass['Sensitivity']), mean(cm_test_b3$byClass['Sensitivity'])),                   # BACC, Sen und Spe
                                'Specificity' = c(mean(cm_train_b3$byClass['Specificity']), mean(cm_test_b3$byClass['Specificity']))))                  # definiert.

  #### Analyse für die Kombination der 3 Blöcke analog zu der in Block 1 ####
  
  # Training des Elastic-Net-Modells für Block 1, 2, 3
enet1_b1_b2_b3 <- train(as.formula(mod_b1_b2_b3), # zuvor spezifiziertes Modell 
                  train_dat[, c(Präd_b1_b2_b3, "replicate")], # Trainingsdaten: Prädiktoren aus Block 1 + "replicate" 
                  metric = "logLoss", # logLoss als Metrik zur Bewertung von Klassifikationsfragestellungen
                  method = "glmnet",  # GLM (generalisiertes lineares Modell) zur Schätzung und Regularisierung
                  family = "binomial", # Binäre Klassifikation
                  
                  # Hyperparametertuning
                  trControl = trainControl(method = "LOOCV", # LOOCV
                                           summaryFunction = mnLogLoss, # mean der logLoss
                                           classProbs = TRUE, # Klassenwahrscheinlichkeiten 
                                           savePredictions = TRUE), # Vorhersagen speichern 
                  
                  # tuneGrid = grid)
                  tuneLength = 21)

# Das anhand der Trainingsdaten trainierte Modell zur Vorhersagend des Testdatensatzes verwenden
pred_train_b1_b2_b3 <- predict(enet1_b1_b2_b3, train_dat) # Trainiertes Modell zur Vorhersage der Trainingsdaten verwenden
pred_test_b1_b2_b3 <- predict(enet1_b1_b2_b3, test_dat) # Trainiertes Modell zur Vorhersage der Testsdaten verwenden

cm_train_b1_b2_b3 <- confusionMatrix(pred_train_b1_b2_b3, train_dat$replicate, positive = "yes") # Konfusionsmatrix Trainingsdaten
cm_test_b1_b2_b3 <- confusionMatrix(pred_test_b1_b2_b3, test_dat$replicate, positive = "yes") # Konfusionsmatrix Testdaten

# Speichern der Ergebnisse jeder Iteration für Block 3
res1_b1_b2_b3 <- rbind(res1_b1_b2_b3, cbind('iter' = i,
                                'mod_b1_b2_b3' = 1:2, 
                                'Balanced Accuracy' = c(mean(cm_train_b1_b2_b3$byClass['Balanced Accuracy']), mean(cm_test_b1_b2_b3$byClass['Balanced Accuracy'])), # Als Output-Variablen
                                'Sensitivity' = c(mean(cm_train_b1_b2_b3$byClass['Sensitivity']), mean(cm_test_b1_b2_b3$byClass['Sensitivity'])),                   # BACC, Sen und Spe
                                'Specificity' = c(mean(cm_train_b1_b2_b3$byClass['Specificity']), mean(cm_test_b1_b2_b3$byClass['Specificity']))))                  # definiert.
}




# ______________________________________________________________________________
#
# Classification and Regression Tree Analyse (CART)
#
# ______________________________________________________________________________

res1_c_b1 <- NULL # Ergbnisobjekte der Prädiktorenblöcke außerhalb des Loops erstellen
res1_c_b2 <- NULL # b1 = Block; b2 = Block 2; b3 = Block 3, b1_b2_b3 = alle Blöcke
res1_c_b3 <- NULL
res1_c_b1_b2_b3 <- NULL


for (i in 1:100) { # Outer Loop mit i = Iterationen
  print(paste0(i, ". Iter"))
  
  set.seed(20190930 + i) # Seed gewährleistet Reproduzierbarkeit der Ergebnisse
  
  trainIndex <- createDataPartition(data_final$replicate, # zufällige Partitionierung des Datensatzes in Trainings- und Testdatensatz
                                    p = .8, # 80% Trainingsdaten; 20% Testdaten
                                    list = FALSE,
                                    times = 1)
  train_dat <- (data_final[ trainIndex, ]) # Speichern der beiden Partitionen
  test_dat  <- (data_final[-trainIndex, ]) #  entfernt alle Missings
  
  # Grid Search wird bei Verfahren mit mehr als einem Hyperparameter vernwendet
  grid <- expand.grid(.cp = seq(.01, .10, .001)) # für minsplit, minbucket & maxdepth werden Defaults verwendet 
  
  
  #### Modelle ####
  
  #### Analyse Prädiktoren Block 1 ####
  
  # Modellspezifikation Block 1
  mod_b1 <- as.formula(paste("replicate ~" , paste(Präd_b1, collapse =" + "))) # Zielvariable "replicate" wird basierend auf den Prädiktoren in Block 1 vorhergesagt
  
  # Training des CART-Modells für Block 1
  cart1_b1 <- train(as.formula(mod_b1), # zuvor spezifiziertes Modell 
                   train_dat[, c(Präd_b1, "replicate")], # Trainingsdaten: Prädiktoren aus Block 1 + "replicate" 
                   metric = "ROC", # ROC bzw. area under the curve (AUC) als Metrik zur Bewertung von Klassifikationsfragestellungen
                   method = "rpart", # Methode für Decision Trees
                   tuneGrid = grid, # zuvor spezifiziertes TuneGrid wählen
                   trControl=trainControl(method = "LOOCV", # LOOCV
                                          classProbs = TRUE, # Klassenwahrscheinlichkeiten
                                          summaryFunction = twoClassSummary, # Metriken für binäre Klassifikation
                                          savePredictions = TRUE), # Vorhersagen speichern
                   na.action = na.pass) # behält Missings und erlaubt Surrogat-Splits
  
  # Das anhand der Trainingsdaten trainierte Modell zur Vorhersagend des Testdatensatzes verwenden
  pred_train_c1_b1 <- predict(cart1_b1, train_dat, na.action = na.pass) # Trainiertes Modell zur Vorhersage der Trainingsdaten verwenden
  pred_test_c1_b1 <- predict(cart1_b1, test_dat, na.action = na.pass) # Trainiertes Modell zur Vorhersage der Testsdaten verwenden
  
  cm_train_c1_b1 <- confusionMatrix(pred_train_c1_b1, train_dat$replicate, positive = "yes") # Konfusionsmatrix Trainingsdaten
  cm_test_c1_b1 <- confusionMatrix(pred_test_c1_b1, test_dat$replicate, positive = "yes") # Konfusionsmatrix Testdaten
  
  # Speichern der Ergebnisse jeder Iteration für Block 1
  res1_c_b1 <- rbind(res1_c_b1, cbind('iter' = i,
                                    'mod_b1' = 1:2,
                                    'Balanced Accuracy' = c(mean(cm_train_c1_b1$byClass['Balanced Accuracy']), mean(cm_test_c1_b1$byClass['Balanced Accuracy'])),
                                    'Sensitivity' = c(mean(cm_train_c1_b1$byClass['Sensitivity']), mean(cm_test_c1_b1$byClass['Sensitivity'])),
                                    'Specificity' = c(mean(cm_train_c1_b1$byClass['Specificity']), mean(cm_test_c1_b1$byClass['Specificity']))))
  
  #### Analyse für Block 2 analog zu der in Block 1 ####
  
  # Modellspezifikation
  mod_b2 <- as.formula(paste("replicate ~" , paste(Präd_b2, collapse =" + ")))   
  
  # Training des CART-Modells
  cart1_b2 <- train(as.formula(mod_b2),
                   train_dat[, c(Präd_b2, "replicate")],
                   metric = "ROC",   
                   method = "rpart",
                   tuneGrid = grid,
                   trControl=trainControl(method = "LOOCV",
                                          classProbs = TRUE,
                                          summaryFunction = twoClassSummary,
                                          savePredictions = TRUE),
                   na.action = na.pass)
  
  # Vorhersage anhand des trainierten Modells
  pred_train_c1_b2 <- predict(cart1_b2, train_dat, na.action = na.pass)
  pred_test_c1_b2 <- predict(cart1_b2, test_dat, na.action = na.pass)
  
  cm_train_c1_b2 <- confusionMatrix(pred_train_c1_b2, train_dat$replicate, positive = "yes")
  cm_test_c1_b2 <- confusionMatrix(pred_test_c1_b2, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res1_c_b2 <- rbind(res1_c_b2, cbind('iter' = i,
                                    'mod_b2' = 1:2,
                                    'Balanced Accuracy' = c(mean(cm_train_c1_b2$byClass['Balanced Accuracy']), mean(cm_test_c1_b2$byClass['Balanced Accuracy'])),
                                    'Sensitivity' = c(mean(cm_train_c1_b2$byClass['Sensitivity']), mean(cm_test_c1_b2$byClass['Sensitivity'])),
                                    'Specificity' = c(mean(cm_train_c1_b2$byClass['Specificity']), mean(cm_test_c1_b2$byClass['Specificity']))))
  
  
  #### Analyse für Block 3 analog zu der in Block 1 ####
  
  # Modellspezifikation
  mod_b3 <- as.formula(paste("replicate ~" , paste(Präd_b3, collapse =" + ")))   
  
  # Training des CART-Modells
  cart1_b3 <- train(as.formula(mod_b3),
                   train_dat[, c(Präd_b3, "replicate")],
                   metric = "ROC",   
                   method = "rpart",
                   tuneGrid = grid,
                   trControl=trainControl(method = "LOOCV",
                                          classProbs = TRUE,
                                          summaryFunction = twoClassSummary,
                                          savePredictions = TRUE),
                   na.action = na.pass)
  
  
  # Vorhersage anhand des trainierten Modells
  pred_train_c1_b3 <- predict(cart1_b3, train_dat, na.action = na.pass)
  pred_test_c1_b3 <- predict(cart1_b3, test_dat, na.action = na.pass)
  
  cm_train_c1_b3 <- confusionMatrix(pred_train_c1_b3, train_dat$replicate, positive = "yes")
  cm_test_c1_b3 <- confusionMatrix(pred_test_c1_b3, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res1_c_b3 <- rbind(res1_c_b3, cbind('iter' = i,
                                    'mod_b3' = 1:2,
                                    'Balanced Accuracy' = c(mean(cm_train_c1_b3$byClass['Balanced Accuracy']), mean(cm_test_c1_b3$byClass['Balanced Accuracy'])),
                                    'Sensitivity' = c(mean(cm_train_c1_b3$byClass['Sensitivity']), mean(cm_test_c1_b3$byClass['Sensitivity'])),
                                    'Specificity' = c(mean(cm_train_c1_b3$byClass['Specificity']), mean(cm_test_c1_b3$byClass['Specificity']))))
  
  
  #### Analyse für die Kombination der 3 Blöcke analog zu der in Block 1 ####
  
  # Modellspezifikation 
  mod_b1_b2_b3 <- as.formula(paste("replicate ~" , paste(Präd_b1_b2_b3, collapse =" + ")))
  
  # Training des CART-Modells
  cart1_b1_b2_b3 <- train(as.formula(mod_b1_b2_b3),
                         train_dat[, c(Präd_b1_b2_b3, "replicate")],
                         metric = "ROC",   
                         method = "rpart",
                         tuneGrid = grid,
                         trControl=trainControl(method = "LOOCV",
                                                classProbs = TRUE,
                                                summaryFunction = twoClassSummary,
                                                savePredictions = TRUE),
                         na.action = na.pass)
  
  
  # Vorhersage anhand des trainierten Modells
  pred_train_c1_b1_b2_b3 <- predict(cart1_b1_b2_b3, train_dat, na.action = na.pass)
  pred_test_c1_b1_b2_b3 <- predict(cart1_b1_b2_b3, test_dat, na.action = na.pass)
  
  cm_train_c1_b1_b2_b3 <- confusionMatrix(pred_train_c1_b1_b2_b3, train_dat$replicate, positive = "yes")
  cm_test_c1_b1_b2_b3 <- confusionMatrix(pred_test_c1_b1_b2_b3, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res1_c_b1_b2_b3 <- rbind(res1_c_b1_b2_b3, cbind('iter' = i,
                                                'mod_b1_b2_b3' = 1:2,
                                                'Balanced Accuracy' = c(mean(cm_train_c1_b1_b2_b3$byClass['Balanced Accuracy']), mean(cm_test_c1_b1_b2_b3$byClass['Balanced Accuracy'])),
                                                'Sensitivity' = c(mean(cm_train_c1_b1_b2_b3$byClass['Sensitivity']), mean(cm_test_c1_b1_b2_b3$byClass['Sensitivity'])),
                                                'Specificity' = c(mean(cm_train_c1_b1_b2_b3$byClass['Specificity']), mean(cm_test_c1_b1_b2_b3$byClass['Specificity']))))
  
}



# ______________________________________________________________________________
#
#  Ergebnisse-ENet für jeden Block
#
# ______________________________________________________________________________


# Auswertung Prädiktorenblock 1

# Block 1: 
(r1_b1  <- res1_b1 %>% data.frame() %>% 
    group_by(mod_b1) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b1))
vip(enet1_b1, num_features = 20)
varImp(enet1_b1, lambda = enet1_b1$lambda.min)
plot(enet1_b1)

coefficients_enet1_b1 <- coef(enet1_b1$finalModel, s = enet1_b1$bestTune$lambda)
print(coefficients_enet1_b1) # Koeffizienten

enet1_b1$bestTune # beste Kombination aus Hyperparametern



# Auswertung Prädiktorenblock 2

(r1_b2  <- res1_b2 %>% data.frame() %>% 
    group_by(mod_b2) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b2))
vip(enet1_b2, num_features = 20)
varImp(enet1_b2, lambda = enet1_b2$lambda.min)
plot(enet1_b2)

coefficients_enet1_b2 <- coef(enet1_b2$finalModel, s = enet1_b2$bestTune$lambda)
print(coefficients_enet1_b2) # Koeffizienten

enet1_b2$bestTune # beste Kombination aus Hyperparametern




# Auswertung Prädiktorenblock 3

(r1_b3  <- res1_b3 %>% data.frame() %>% 
    group_by(mod_b3) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b3))
vip(enet1_b3, num_features = 20)
varImp(enet1_b3, lambda = enet1_b3$lambda.min)
plot(enet1_b3)

coefficients_enet1_b3 <- coef(enet1_b3$finalModel, s = enet1_b3$bestTune$lambda)
print(coefficients_enet1_b3) # Koeffizienten

enet1_b3$bestTune # beste Kombination aus Hyperparametern



# Auswertung Prädiktorenblock 1,2 und 3

(r1_b1_b2_b3  <- res1_b1_b2_b3 %>% data.frame() %>% 
    group_by(mod_b1_b2_b3) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b1_b2_b3))
vip(enet1_b1_b2_b3, num_features = 20)
varImp(enet1_b1_b2_b3, lambda = enet1_b1_b2_b3$lambda.min)
plot(enet1_b1_b2_b3)

coefficients_enet1_b1_b2_b3 <- coef(enet1_b1_b2_b3$finalModel, s = enet1_b1_b2_b3$bestTune$lambda)
print(coefficients_enet1_b1_b2_b3) # Koeffizienten

enet1_b1_b2_b3$bestTune # beste Kombination aus Hyperparametern



# ______________________________________________________________________________
#
# Ergebnisse CART für jeden Block
#
# ______________________________________________________________________________

# Auswertung Prädiktorenblock 1

(r1_c_b1 <- res1_c_b1%>% data.frame() %>%
  group_by(mod_b1) %>%
  summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                     sd = ~sd(., na.rm = TRUE))) %>% 
  round(., 3) %>% rename(Split = mod_b1))

vip(cart1_b1, num_features = 20)
plot(cart1_b1)

cart1_b1$finalModel$cp
cart1_b1$finalModel$control$minsplit
cart1_b1$finalModel$control$minbucket
cart1_b1$finalModel$control$maxdepth

rpart.plot::rpart.plot(cart1_b1$finalModel,
                       fallen.leaves = TRUE,
                       box.palette = "RdGn")



# Auswertung Prädiktorenblock 2

(r1_c_b2 <- res1_c_b2%>% data.frame() %>%
    group_by(mod_b2) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b2))

vip(cart1_b2, num_features = 20)

plot(cart1_b2)

cart1_b2$finalModel$cp
cart1_b2$finalModel$control$minsplit
cart1_b2$finalModel$control$minbucket
cart1_b2$finalModel$control$maxdepth

rpart.plot::rpart.plot(cart1_b2$finalModel,
                       fallen.leaves = TRUE,
                       box.palette = "RdGn")



# Auswertung Prädiktorenblock 3

(r1_c_b3 <- res1_c_b3%>% data.frame() %>%
    group_by(mod_b3) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b3))

vip(cart1_b3, num_features = 20)
plot(cart1_b3)

cart1_b3$finalModel$cp
cart1_b3$finalModel$control$minsplit
cart1_b3$finalModel$control$minbucket
cart1_b3$finalModel$control$maxdepth

rpart.plot::rpart.plot(cart1_b3$finalModel,
                       fallen.leaves = TRUE,
                       box.palette = "RdGn")



# Auswertung Prädiktorenblock 1,2 und 3

(r_c1_b1_b2_b3 <- res1_c_b1_b2_b3%>% data.frame() %>%
    group_by(mod_b1_b2_b3) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b1_b2_b3))

vip(cart1_b1_b2_b3, num_features = 20)
plot(cart1_b1_b2_b3)

cart1_b1_b2_b3$finalModel$cp
cart1_b1_b2_b3$finalModel$control$minsplit
cart1_b1_b2_b3$finalModel$control$minbucket
cart1_b1_b2_b3$finalModel$control$maxdepth

rpart.plot::rpart.plot(cart1_b1_b2_b3$finalModel,
                       fallen.leaves = TRUE,
                       box.palette = "RdGn")
