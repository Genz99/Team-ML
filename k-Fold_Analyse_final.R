# ______________________________________________________________________________
#
# Logistische Elastic-Net Regression (E-Net)
#
# ______________________________________________________________________________


res_b1 <- NULL # Ergbnisobjekte der Prädiktorenblöcke außerhalb des Loops erstellen
res_b2 <- NULL # b1 = Block; b2 = Block 2; b3 = Block 3, b1_b2_b3 = alle Blöcke
res_b3 <- NULL
res_b1_b2_b3 <- NULL

coef_b1 <- NULL # Objekt für Koeffizienten 


for (i in 1:100) { # Outer Loop 
  print(paste0(i, ". Iter"))
  
  set.seed(20190930+i) # Seed gewährleistet Reproduzierbarkeit der Ergebnisse
  
  trainIndex <- createDataPartition(data_final$replicate, # zufällige Partitionierung des Datensatzes in Trainings- und Testdatensatz
                                    p = .8, # 80% Trainingsdaten; 20% Testdaten
                                    list = FALSE,
                                    times = 1)
  train_dat <- (data_final[ trainIndex, ]) # Speichern der beiden Partitionen
  test_dat  <- (data_final[-trainIndex, ]) #  entfernt alle Missings
  

  # Pre-Processing: Transformation der Daten nach Partitionierung um Data-Leakage zu vermeiden
  preProcValues_train <- preProcess(train_dat, method = c("knnImpute")) # Trainingsdaten: knn-Impute (gleichzeitig z-Standardisiert)
  train_dat <- predict(preProcValues_train, train_dat) # speichern der z-Standardisierten Trainingsdaten
  
  preProcValues_test <- preProcess(test_dat, method = c("knnImpute")) # Testdaten: knn-Impute (gleichzeitig z-Standardisiert)
  test_dat <- predict(preProcValues_test, test_dat) # speichern der z-Standardisierten Testsdaten
  

  # grid <- expand.grid(alpha = seq(0, 1, length = 11),
  #                      lambda = seq(0.001, 0.1, length = 21)) # Art des Hyperparametertunings

  #### Modelle ####
  
  #### Analyse Prädiktoren Block 1 ####
  
  # Modellspezifikation Block 1
  mod_b1 <- as.formula(paste("replicate ~" , paste(Präd_b1, collapse =" + "))) # Zielvariable "replicate" wird basierend auf den Prädiktoren in Block 1 vorhergesagt
  
  # Training des Elastic-Net-Modells für Block 1
  enet_b1 <- train(as.formula(mod_b1), # zuvor spezifiziertes Modell 
                   train_dat[, c(Präd_b1, "replicate")], # Trainingsdaten: Prädiktoren aus Block 1 + "replicate" 
                   metric = "logLoss", # logLoss als Metrik zur Bewertung von Klassifikationsfragestellungen
                   method = "glmnet",  # GLM (generalisiertes lineares Modell) zur Schätzung und Regularisierung
                   family = "binomial", # Binäre Klassifikation
                   
                   # Hyperparametertuning
                   trControl = trainControl(method = "cv", # k-fold Cross-Validation
                                            number = 10, # k = 10
                                            summaryFunction = mnLogLoss, # mean der logLoss
                                            classProbs = TRUE, # Klassenwahrscheinlichkeiten 
                                            savePredictions = TRUE), # Vorhersagen speichern 
                   
                   tuneLength = 21)
                   # tuneGrid = grid)
  
  
  # Das anhand der Trainingsdaten trainierte Modell zur Vorhersagend des Testdatensatzes verwenden
  pred_train_b1 <- predict(enet_b1, train_dat) # Trainiertes Modell zur Vorhersage der Trainingsdaten verwenden
  pred_test_b1 <- predict(enet_b1, test_dat) # Trainiertes Modell zur Vorhersage der Testsdaten verwenden
  
  cm_train_b1 <- confusionMatrix(pred_train_b1, train_dat$replicate, positive = "yes") # Konfusionsmatrix Trainingsdaten
  cm_test_b1 <- confusionMatrix(pred_test_b1, test_dat$replicate, positive = "yes") # Konfusionsmatrix Testdaten
  
  # Speichern der Ergebnisse jeder Iteration für Block 1
  res_b1 <- rbind(res_b1, cbind('iter' = i,
                                'mod_b1' = 1:2, 
                                'Balanced Accuracy' = c(mean(cm_train_b1$byClass['Balanced Accuracy']), mean(cm_test_b1$byClass['Balanced Accuracy'])), # Als Output-Variablen
                                'Sensitivity' = c(mean(cm_train_b1$byClass['Sensitivity']), mean(cm_test_b1$byClass['Sensitivity'])),                   # BACC, Sen und Spe
                                'Specificity' = c(mean(cm_train_b1$byClass['Specificity']), mean(cm_test_b1$byClass['Specificity']))))                  # definiert.
 
  # Koeffizienten extrahieren
  coefs <- as.numeric(as.matrix(coef(enet_b1$finalModel, s = enet_b1$bestTune$lambda)))  # Extrahieren der finalen Koeffizienten
  
  # Speichern der Koeffizienten in der Matrix
  coef_b1 <- rbind(coef_b1, coefs)
  
 
  #### Analyse für Block 2 analog zu der in Block 1 ####
  
  # Modellspezifikation
  mod_b2 <- as.formula(paste("replicate ~", paste(Präd_b2, collapse = " + ")))
  
  # Training des Elastic-Net-Modells
  enet_b2 <- train(mod_b2,
                   train_dat[, c(Präd_b2, "replicate")],
                   metric = "logLoss",
                   method = "glmnet",
                   family = "binomial",
                   
                   # Hyperparametertuning
                   trControl = trainControl(method = "cv",
                                            number = 10,
                                            summaryFunction = mnLogLoss,
                                            classProbs = TRUE,
                                            savePredictions = TRUE),
                   tuneLength = 21)
                   # tuneGrid = grid)
  
  # Vorhersage anhand des trainierten Modells
  pred_train_b2 <- predict(enet_b2, train_dat)
  pred_test_b2 <- predict(enet_b2, test_dat)
  
  cm_train_b2 <- confusionMatrix(pred_train_b2, train_dat$replicate, positive = "yes")
  cm_test_b2 <- confusionMatrix(pred_test_b2, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res_b2 <- rbind(res_b2, cbind('iter' = i,
                                'mod_b2' = 1:2,
                                'Balanced Accuracy' = c(mean(cm_train_b2$byClass['Balanced Accuracy']), mean(cm_test_b2$byClass['Balanced Accuracy'])),
                                'Sensitivity' = c(mean(cm_train_b2$byClass['Sensitivity']), mean(cm_test_b2$byClass['Sensitivity'])),
                                'Specificity' = c(mean(cm_train_b2$byClass['Specificity']), mean(cm_test_b2$byClass['Specificity']))))
  
  
  #### Analyse für Block 2 analog zu der in Block 1 ####
  
  # Modellspezifikation
  mod_b3 <- as.formula(paste("replicate ~", paste(Präd_b3, collapse = " + ")))
  
  # Training des Elastic-Net-Modells
  enet_b3 <- train(mod_b3,
                   train_dat[, c(Präd_b3, "replicate")],
                   metric = "logLoss",
                   method = "glmnet",
                   family = "binomial",
                   trControl = trainControl(method = "cv",
                                            number = 10,
                                            summaryFunction = mnLogLoss,
                                            classProbs = TRUE,
                                            savePredictions = TRUE),
                   tuneLength = 21)
                   # tuneGrid = grid)
  
  # Vorhersage anhand des trainierten Modells
  pred_train_b3 <- predict(enet_b3, train_dat)
  pred_test_b3 <- predict(enet_b3, test_dat)
  
  cm_train_b3 <- confusionMatrix(pred_train_b3, train_dat$replicate, positive = "yes")
  cm_test_b3 <- confusionMatrix(pred_test_b3, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res_b3 <- rbind(res_b3, cbind('iter' = i,
                                'mod_b3' = 1:2,
                                'Balanced Accuracy' = c(mean(cm_train_b3$byClass['Balanced Accuracy']), mean(cm_test_b3$byClass['Balanced Accuracy'])),
                                'Sensitivity' = c(mean(cm_train_b3$byClass['Sensitivity']), mean(cm_test_b3$byClass['Sensitivity'])),
                                'Specificity' = c(mean(cm_train_b3$byClass['Specificity']), mean(cm_test_b3$byClass['Specificity']))))
  
  
  #### Analyse für die Kombination der 3 Blöcke analog zu der in Block 1 ####
  
  # Modellspezifikation
  mod_b1_b2_b3 <- as.formula(paste("replicate ~" , paste(Präd_b1_b2_b3, collapse =" + ")))   
  
  # Training des Elastic-Net-Modells
  enet_b1_b2_b3 <- train(as.formula(mod_b1_b2_b3),
                   train_dat[, c(Präd_b1_b2_b3, "replicate")],
                   metric = "logLoss", 
                   method = "glmnet",  
                   family = "binomial",
                   
                   # Hyperparametertuning
                   trControl = trainControl(method = "cv",
                                            number = 10,
                                            summaryFunction = mnLogLoss,
                                            classProbs = TRUE,
                                            savePredictions = TRUE),
                   tuneLength = 21)
                   # tuneGrid = grid)
  
  # Vorhersage anhand des trainierten Modells
  pred_train_b1_b2_b3 <- predict(enet_b1_b2_b3, train_dat)
  pred_test_b1_b2_b3 <- predict(enet_b1_b2_b3, test_dat)
  
  cm_train_b1_b2_b3 <- confusionMatrix(pred_train_b1_b2_b3, train_dat$replicate, positive = "yes")
  cm_test_b1_b2_b3 <- confusionMatrix(pred_test_b1_b2_b3, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res_b1_b2_b3 <- rbind(res_b1_b2_b3, cbind('iter' = i,
                                'mod_b1_b2_b3' = 1:2,
                                'Balanced Accuracy' = c(mean(cm_train_b1_b2_b3$byClass['Balanced Accuracy']), mean(cm_test_b1_b2_b3$byClass['Balanced Accuracy'])),
                                'Sensitivity' = c(mean(cm_train_b1_b2_b3$byClass['Sensitivity']), mean(cm_test_b1_b2_b3$byClass['Sensitivity'])),
                                'Specificity' = c(mean(cm_train_b1_b2_b3$byClass['Specificity']), mean(cm_test_b1_b2_b3$byClass['Specificity']))))
}



# ______________________________________________________________________________
#
# Classification and Regression Trees Analyse (CART)
#
# ______________________________________________________________________________


res_c_b1 <- NULL # Ergbnisobjekte der Prädiktorenblöcke außerhalb des Loops erstellen
res_c_b2 <- NULL # b1 = Block; b2 = Block 2; b3 = Block 3, b1_b2_b3 = alle Blöcke
res_c_b3 <- NULL
res_c_b1_b2_b3 <- NULL


for (i in 1:100) { # Outer Loop 
  print(paste0(i, ". Iter"))
  
  set.seed(20190930+i) # Seed gewährleistet Reproduzierbarkeit der Ergebnisse
  
  trainIndex <- createDataPartition(data_final$replicate, # zufällige Partitionierung des Datensatzes in Trainings- und Testdatensatz
                                    p = .8, # 80% Trainingsdaten; 20% Testdaten
                                    list = FALSE,
                                    times = 1)
  train_dat <- (data_final[ trainIndex, ]) # Speichern der beiden Partitionen
  test_dat  <- (data_final[-trainIndex, ]) #  entfernt alle Missings
  
  
  # Grid Search wird bei Verfahren mit mehr als einem Hyperparameter vernwendet
  grid <- expand.grid(.cp = seq(.01, .10, .001))
  
  
  #### Modelle ####
  
  #### Analyse Prädiktoren Block 1 ####
  
  # Modellspezifikation Block 1
  mod_b1 <- as.formula(paste("replicate ~" , paste(Präd_b1, collapse =" + "))) # Zielvariable "replicate" wird basierend auf den Prädiktoren in Block 1 vorhergesagt
  
  # Training des CART-Modells für Block 1
  cart_b1 <- train(as.formula(mod_b1), # zuvor spezifiziertes Modell 
                   train_dat[, c(Präd_b1, "replicate")], # Trainingsdaten: Prädiktoren aus Block 1 + "replicate" 
                   metric = "ROC", # ROC bzw. area under the curve (AUC) als Metrik zur Bewertung von Klassifikationsfragestellungen
                   method = "rpart", # Methode für Decision Trees
                   tuneGrid = grid, # zuvor spezifiziertes TuneGrid wählen
                   
                   trControl=trainControl(method = "cv", # k-fold Cross-Validation
                                          number = 10, # k = 10
                                          classProbs = TRUE, # Klassenwahrscheinlichkeiten
                                          summaryFunction = twoClassSummary, # Metriken für binäre Klassifikation
                                          savePredictions = TRUE),
                                          # search = "random"), # Random Search
                   # tuneLength = 210,
                   na.action = na.pass) # behält Missings und erlaubt Surrogat-Splits
  
  
  # Das anhand der Trainingsdaten trainierte Modell zur Vorhersagend des Testdatensatzes verwenden
  pred_train_c_b1 <- predict(cart_b1, train_dat, na.action = na.pass) # Trainiertes Modell zur Vorhersage der Trainingsdaten verwenden
  pred_test_c_b1 <- predict(cart_b1, test_dat, na.action = na.pass) # Trainiertes Modell zur Vorhersage der Testsdaten verwenden
  
  cm_train_c_b1 <- confusionMatrix(pred_train_c_b1, train_dat$replicate, positive = "yes") # Konfusionsmatrix Trainingsdaten
  cm_test_c_b1 <- confusionMatrix(pred_test_c_b1, test_dat$replicate, positive = "yes") # Konfusionsmatrix Testdaten
  
  # Speichern der Ergebnisse jeder Iteration für Block 1
  res_c_b1 <- rbind(res_c_b1, cbind('iter' = i,
                                'mod_b1' = 1:2,
                                'Balanced Accuracy' = c(mean(cm_train_c_b1$byClass['Balanced Accuracy']), mean(cm_test_c_b1$byClass['Balanced Accuracy'])),
                                'Sensitivity' = c(mean(cm_train_c_b1$byClass['Sensitivity']), mean(cm_test_c_b1$byClass['Sensitivity'])),
                                'Specificity' = c(mean(cm_train_c_b1$byClass['Specificity']), mean(cm_test_c_b1$byClass['Specificity']))))
  
  #### Analyse für Block 2 analog zu der in Block 1 ####
  
  # Modellspezifikation
  mod_b2 <- as.formula(paste("replicate ~" , paste(Präd_b2, collapse =" + ")))   
  
  # Training des CART-Modells
  cart_b2 <- train(as.formula(mod_b2),
                   train_dat[, c(Präd_b2, "replicate")],
                   metric = "ROC",   
                   method = "rpart",
                   tuneGrid = grid,
                   
                   trControl=trainControl(method = "cv",
                                          number = 10,
                                          classProbs = TRUE,
                                          summaryFunction = twoClassSummary,
                                          savePredictions = TRUE),
                                          # search = "random"),
                   # tuneLength = 210,
                   na.action = na.pass)
  
  # Vorhersage anhand des trainierten Modells
  pred_train_c_b2 <- predict(cart_b2, train_dat, na.action = na.pass)
  pred_test_c_b2 <- predict(cart_b2, test_dat, na.action = na.pass)
  
  cm_train_c_b2 <- confusionMatrix(pred_train_c_b2, train_dat$replicate, positive = "yes")
  cm_test_c_b2 <- confusionMatrix(pred_test_c_b2, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res_c_b2 <- rbind(res_c_b2, cbind('iter' = i,
                                    'mod_b2' = 1:2,
                                    'Balanced Accuracy' = c(mean(cm_train_c_b2$byClass['Balanced Accuracy']), mean(cm_test_c_b2$byClass['Balanced Accuracy'])),
                                    'Sensitivity' = c(mean(cm_train_c_b2$byClass['Sensitivity']), mean(cm_test_c_b2$byClass['Sensitivity'])),
                                    'Specificity' = c(mean(cm_train_c_b2$byClass['Specificity']), mean(cm_test_c_b2$byClass['Specificity']))))
  
  
  #### Analyse für Block 3 analog zu der in Block 1 ####
  
  # Modellspezifikation
  mod_b3 <- as.formula(paste("replicate ~" , paste(Präd_b3, collapse =" + ")))   
  
  # Training des CART-Modells
  cart_b3 <- train(as.formula(mod_b3),
                   train_dat[, c(Präd_b3, "replicate")],
                   metric = "ROC",   
                   method = "rpart",
                   tuneGrid = grid,
                   
                   trControl=trainControl(method = "cv",
                                          number = 10,
                                          classProbs = TRUE,
                                          summaryFunction = twoClassSummary,
                                          savePredictions = TRUE),
                                          # search = "random"),
                   # tuneLength = 210,
                   na.action = na.pass)
  
  
  # Vorhersage anhand des trainierten Modells
  pred_train_c_b3 <- predict(cart_b3, train_dat, na.action = na.pass)
  pred_test_c_b3 <- predict(cart_b3, test_dat, na.action = na.pass)
  
  cm_train_c_b3 <- confusionMatrix(pred_train_c_b3, train_dat$replicate, positive = "yes")
  cm_test_c_b3 <- confusionMatrix(pred_test_c_b3, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res_c_b3 <- rbind(res_c_b3, cbind('iter' = i,
                                    'mod_b3' = 1:2,
                                    'Balanced Accuracy' = c(mean(cm_train_c_b3$byClass['Balanced Accuracy']), mean(cm_test_c_b3$byClass['Balanced Accuracy'])),
                                    'Sensitivity' = c(mean(cm_train_c_b3$byClass['Sensitivity']), mean(cm_test_c_b3$byClass['Sensitivity'])),
                                    'Specificity' = c(mean(cm_train_c_b3$byClass['Specificity']), mean(cm_test_c_b3$byClass['Specificity']))))
  
  
  #### Analyse für die Kombination der 3 Blöcke analog zu der in Block 1 ####
  
  # Modellspezifikation 
  mod_b1_b2_b3 <- as.formula(paste("replicate ~" , paste(Präd_b1_b2_b3, collapse =" + ")))
  
  # Training des CART-Modells
  cart_b1_b2_b3 <- train(as.formula(mod_b1_b2_b3),
                         train_dat[, c(Präd_b1_b2_b3, "replicate")],
                         metric = "ROC",   
                         method = "rpart",
                         tuneGrid = grid,
                         
                         trControl=trainControl(method = "cv",
                                                number = 10,
                                                classProbs = TRUE,
                                                summaryFunction = twoClassSummary,
                                                savePredictions = TRUE),
                                                # search = "random"),
                         # tuneLength = 210,
                         na.action = na.pass)
  
  
  # Vorhersage anhand des trainierten Modells
  pred_train_c_b1_b2_b3 <- predict(cart_b1_b2_b3, train_dat, na.action = na.pass)
  pred_test_c_b1_b2_b3 <- predict(cart_b1_b2_b3, test_dat, na.action = na.pass)
  
  cm_train_c_b1_b2_b3 <- confusionMatrix(pred_train_c_b1_b2_b3, train_dat$replicate, positive = "yes")
  cm_test_c_b1_b2_b3 <- confusionMatrix(pred_test_c_b1_b2_b3, test_dat$replicate, positive = "yes")
  
  # Speichern der Ergebnisse jeder Iteration
  res_c_b1_b2_b3 <- rbind(res_c_b1_b2_b3, cbind('iter' = i,
                                    'mod_b1_b2_b3' = 1:2,
                                    'Balanced Accuracy' = c(mean(cm_train_c_b1_b2_b3$byClass['Balanced Accuracy']), mean(cm_test_c_b1_b2_b3$byClass['Balanced Accuracy'])),
                                    'Sensitivity' = c(mean(cm_train_c_b1_b2_b3$byClass['Sensitivity']), mean(cm_test_c_b1_b2_b3$byClass['Sensitivity'])),
                                    'Specificity' = c(mean(cm_train_c_b1_b2_b3$byClass['Specificity']), mean(cm_test_c_b1_b2_b3$byClass['Specificity']))))
  
}


# ______________________________________________________________________________
#
# Über die Iterationen gemittelte Ergebnisse-ENet für jeden Block
#
# ______________________________________________________________________________

# Block 1: 
(r_b1  <- res_b1 %>% data.frame() %>% 
    group_by(mod_b1) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b1))

vip(enet_b1, num_features = 20) 

varImp(enet_b1, lambda = enet_b1$lambda.min)

coefficients_enet_b1 <- coef(enet_b1$finalModel, s = enet_b1$bestTune$lambda)
print(coefficients_enet_b1) # Koeffizienten

enet_b1$bestTune # Hyperparameter


# Block 2
(r_b2  <- res_b2 %>% data.frame() %>%
    group_by(mod_b2) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b2))

vip(enet_b2, num_features = 20)

varImp(enet_b2, lambda = enet_b2$lambda.min)

coefficients_enet_b2 <- coef(enet_b2$finalModel, s = enet_b2$bestTune$lambda)
print(coefficients_enet_b2) # Koeffizienten

enet_b2$bestTune # Hyperparameter


# Block 3
(r_b3  <- res_b3 %>% data.frame() %>%
    group_by(mod_b3) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b3))

vip(enet_b3, num_features = 20)

varImp(enet_b3, lambda = enet_b3$lambda.min)

coefficients_enet_b3 <- coef(enet_b3$finalModel, s = enet_b3$bestTune$lambda)
print(coefficients_enet_b3) # Koeffizienten

enet_b3$bestTune # Hyperparameter


# Alle Blöcke 
(r_b1_b2_b3  <- res_b1_b2_b3 %>% data.frame() %>%
    group_by(mod_b1_b2_b3) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b1_b2_b3))

vip(enet_b1_b2_b3, num_features = 20)

varImp(enet_b1_b2_b3, lambda = enet_b1_b2_b3$lambda.min)

coefficients_enet_b1_b2_b3 <- coef(enet_b1_b2_b3$finalModel, s = enet_b1_b2_b3$bestTune$lambda)
print(coefficients_enet_b1_b2_b3) # Koeffizinten als Logit 
print(exp(coefficients_enet_b1_b2_b3)) # Koeffizienten als Odds-Ratios

enet_b1_b2_b3$bestTune # Hyperparameter


# ______________________________________________________________________________
#
# Über die Iterationen gemittelte Ergebnisse CART für jeden Block
#
# ______________________________________________________________________________

# Block 1
(r_c_b1 <- res_c_b1%>% data.frame() %>%
    group_by(mod_b1) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b1))

vip(cart_b1, num_features = 20)

rpart.plot(cart_b1$finalModel, # Visualisierung
           fallen.leaves = TRUE,
           box.palette = "GnRd")

# Hyperparameter
cart_b1$finalModel$cp
cart_b1$finalModel$control$minsplit
cart_b1$finalModel$control$minbucket
cart_b1$finalModel$control$maxdepth   


# Block 2
(r_c_b2 <- res_c_b2%>% data.frame() %>%
    group_by(mod_b2) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b2))

vip(cart_b2, num_features = 20)

rpart.plot(cart_b2$finalModel, # Visualisierung
           fallen.leaves = TRUE,
           box.palette = "GnRd")

# Hyperparameter
cart_b2$finalModel$cp  
cart_b2$finalModel$control$minsplit
cart_b2$finalModel$control$minbucket
cart_b2$finalModel$control$maxdepth   


# Block 3
(r_c_b3 <- res_c_b3%>% data.frame() %>%
    group_by(mod_b3) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b3))

vip(cart_b3, num_features = 20)

rpart.plot(cart_b3$finalModel, # Visualisierung
           fallen.leaves = TRUE,
           box.palette = "GnRd")

# Hyperparameter
cart_b3$finalModel$cp
cart_b3$finalModel$control$minsplit
cart_b3$finalModel$control$minbucket
cart_b3$finalModel$control$maxdepth   


# Alle Blöcke 
(r_c_b1_b2_b3 <- res_c_b1_b2_b3%>% data.frame() %>%
    group_by(mod_b1_b2_b3) %>%
    summarize_all(list(mean = ~mean(., na.rm = TRUE), 
                       sd = ~sd(., na.rm = TRUE))) %>% 
    round(., 3) %>% rename(Split = mod_b1_b2_b3))

vip(cart_b1_b2_b3, num_features = 20)

rpart.plot(cart_b1_b2_b3$finalModel, # Visualisierung
           fallen.leaves = TRUE,
           box.palette = "GnRd")

# Hyperparameter
cart_b1_b2_b3$finalModel$cp
cart_b1_b2_b3$finalModel$control$minsplit
cart_b1_b2_b3$finalModel$control$minbucket
cart_b1_b2_b3$finalModel$control$maxdepth   


# ______________________________________________________________________________
#
# Tabelle mit allen Ergebnisse 
# 
# ______________________________________________________________________________


r <- NULL
r <- rbind(r_b1, r_c_b1, r_b2, r_c_b2, r_b3, r_c_b3, r_b1_b2_b3, r_c_b1_b2_b3) 
r$Split <- ifelse(r$Split == 1, "Train", "Test")
r$Mod <- c("Enet_b1", "Enet_b1","CART_b1","CART_b1", 
           "Enet_b2", "Enet_b2","CART_b2","CART_b2",
           "Enet_b3", "Enet_b3","CART_b3","CART_b3", 
           "Enet_b1_b2_b3", "Enet_b1_b2_b3", "CART_b1_b2_b3", "CART_b1_b2_b3") 
r <- r[, c(ncol(r), 1:(ncol(r) - 1))]

print(r)

#_______________________________________________________________________________
#
# Odds-Ratios; Block 1; 10-Fold E-Net
#
# ______________________________________________________________________________

OR_b1 <- exp(coef_b1)
coef_names <- coef(enet_b1$finalModel, s = enet_b1$bestTune$lambda)
colnames(OR_b1) <- coef_names@Dimnames[[1]]
rownames(OR_b1) <- NULL

# Violinen Plot

violin_data <- data.frame(OR_b1)
rownames(violin_data) <- NULL

violin_long <- violin_data %>% 
  pivot_longer(cols = everything(), names_to = "Koeffizient", values_to = "OR")

# Violin Plot
ggplot(violin_long, aes(x = Koeffizient, y = OR)) +
  geom_violin(fill = "#BA7F64", alpha = 0.5, adjust = 1, scale = 3) +  
  geom_point(size = 0.5, color = "black") +                     
  geom_hline(yintercept = 1, color = "black", linetype = "dashed", size = 0.5) +
  # geom_boxplot(width = 0.1, color = "black", alpha = 0.5) +  
  theme_minimal() +  
  coord_flip() +  
  scale_y_continuous(breaks = seq(0,6,1))
labs(title = " ", x = "", y = " ") +
  theme(axis.text = element_text(size = 12))  


# Mean and SD
df_OR <- NULL

for (i in (1:ncol(OR_b1))) {
  
  Koeffizient <- colnames(OR_b1)
  
  Mean <- mean(OR_b1[,i])
  
  SD <- SD(OR_b1[,i])
  
  UP <- Mean + 1.96 * SD
  
  LOW <- Mean - 1.96 * SD
  
  df_OR <- data.frame(rbind(df_OR, rbind(c("Koeffizient" = Koeffizient[i],
                                           "Mean" = round(Mean, digits = 2),
                                           "SD" = round(SD, digits = 2), 
                                           "UP" = round(UP, digits = 2),
                                           "LOW" = round(LOW, digits = 2)))))
}

df_OR$Koeffizient <- c("Intercept", "2 bis 3 Autor*Innen", "mehr als 3 Autor*Innen",
                       "Zitationen Erstautor*In", "Überraschende Ergebnisse", "Anzahl konzeptueller 
                       Replikationen", 
                       "p-Value < 0.5", "p-Value > 0.5", "Interaktionseffekt",
                       "sonstiger Effekt", "min Power Quotient", "Effektgröße")

df_OR$Mean <- as.numeric(df_OR$Mean)
df_OR$SD <- as.numeric(df_OR$SD)
df_OR$UP <- as.numeric(df_OR$UP)
df_OR$LOW <- as.numeric(df_OR$LOW)


# Plot
ggplot(df_OR, aes(x = factor(Koeffizient, levels = Koeffizient), y = Mean)) +
  geom_point(size = 2, color = "#BA7F64") +                     
  geom_hline(yintercept = 1, color = "black", linetype = "dashed", size = 0.5) +
  geom_errorbar(aes(ymin = LOW, ymax = UP),       
                width = 0.2, color = "black") +
  scale_y_continuous(breaks = seq(-2,4,1)) +
  coord_flip() +  
  # theme(text = element_text(size = 30)) +
  theme(axis.text.x = element_text(size = 14)) +
  theme_minimal() + 
  labs(title = "",
       x = "", y = "Odds-Ratio")
