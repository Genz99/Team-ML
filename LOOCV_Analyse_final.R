# ______________________________________________________________________________
#
# Logistische Elastic-Net Regression (E-Net)
#
# ______________________________________________________________________________

res1_b1 <- NULL # Ergbnisobjekte der Prädiktorenblöcke außerhalb des Loops erstellen
res1_b2 <- NULL # b1 = Block; b2 = Block 2; b3 = Block 3, b1_b2_b3 = alle Blöcke
res1_b3 <- NULL
res1_b1_b2_b3 <- NULL



for (i in 1:1) { # Outer Loop mit einer Iteration, um alle E-Net Modelle zu schätzen
  print(paste0(i, ". Iter"))
  
  set.seed(20190930 + i) # Seed gewährleistet Reproduzierbarkeit der Ergebnisse
  data_final <- na.omit(data_final) # Entfernen von NA-Werten aus den Daten
  
  train_control_loocv_sd <- trainControl(method = "LOOCV", # Definieren der Leave-on-out Cross Validation
                                         preProc = c("center", "scale"), # standardisieren der Daten für jede Iteration der Cross Validation
                                         summaryFunction = mnLogLoss, # mean der logLoss
                                         classProbs = TRUE, # Klassenwahrscheinlichkeiten 
                                         savePredictions = TRUE) # Vorhersagen speichern 
 
  grid <- expand.grid(alpha = seq(0, 1, length = 11),        # GridSearch um Hyperparameter zu optimieren
                      lambda = seq(0.001, 0.1, length = 21)) # verschiede Sequenzen getestet (siehe Anhang)
 
  
  #### Modelle ####
  
  #### Analyse Prädiktoren Block 1 ####
  
  # Training des Elastic-Net-Modells für Block 1
  enet1_b1 <- train(mod_b1, # zuvor spezifiziertes Modell 
                    data_final[, c(Präd_b1, "replicate")], # Trainingsdaten: Prädiktoren aus Block 1 + "replicate" 
                    metric = "logLoss", # logLoss als Metrik zur Bewertung von Klassifikationsfragestellungen
                    method = "glmnet", # GLM (generalisiertes lineares Modell) zur Schätzung und Regularisierung
                    family = "binomial", # Binäre Klassifikation
                    trControl = train_control_loocv_sd, # Definieren der Art der Cross Validation (s.o.)
                    tuneGrid = grid # Hyperparametertuning von alpha und lambda
                    # tuneLength = 21  -> als Alternative für GridSearch genutzt, alternativ auch tuneLength = 50 (random search)
                    )
  
  preds <- predict(enet1_b1, data_final) # Definieren notwendiger Objekte für das
  obs <- data_final$replicate            # Erstellen der Confusion Matrix
  
  cm <- caret::confusionMatrix(preds, obs) # Erstellen einer Confusion Matrix zur Berechnung der Balanced Accuracy
  balanced_acc <- cm$byClass["Balanced Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
 
  # Abspeichern der Ergebnisse im Ergebnisobjekt 
  res_enet1_b1 <- postResample(pred = preds, obs = obs)
  res_enet1_b1 <- c(res_enet1_b1, Balanced_Accuracy = balanced_acc, Sensitivity = sensitivity, Specificity = specificity)
  
  res1_b1 <- rbind(res1_b1, cbind('iter' = i, 'mod_b1' = 1, res_enet1_b1))
  
  
  #### Analyse für Block 2 analog zu der in Block 1 ####
  
  # Training des Elastic-Net-Modells für Block 2
  enet1_b2 <- train(mod_b2,
                    data_final[, c(Präd_b2, "replicate")],
                    metric = "logLoss",
                    method = "glmnet",
                    family = "binomial",
                    trControl = train_control_loocv_sd,
                    tuneGrid = grid) 
  
  preds <- predict(enet1_b2, data_final)
  obs <- data_final$replicate
  
  cm <- caret::confusionMatrix(preds, obs)
  balanced_acc <- cm$byClass["Balanced Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
 
  # Abspeichern der Ergebnisse im Ergebnisobjekt  
  res_enet1_b2 <- postResample(pred = preds, obs = obs)
  res_enet1_b2 <- c(res_enet1_b2, Balanced_Accuracy = balanced_acc, Sensitivity = sensitivity, Specificity = specificity)
  
  res1_b2 <- rbind(res1_b2, cbind('iter' = i, 'mod_b2' = 1, res_enet1_b2))
  
  
  #### Analyse für Block 3 analog zu der in Block 1 ####
  
  # Training des Elastic-Net-Modells für Block 3
  enet1_b3 <- train(mod_b3,
                    data_final[, c(Präd_b3, "replicate")],
                    metric = "logLoss",
                    method = "glmnet",
                    family = "binomial",
                    trControl = train_control_loocv_sd,
                    tuneGrid = grid) 
  
  preds <- predict(enet1_b3, data_final)
  obs <- data_final$replicate
  
  cm <- caret::confusionMatrix(preds, obs)
  balanced_acc <- cm$byClass["Balanced Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # Abspeichern der Ergebnisse im Ergebnisobjekt 
  res_enet1_b3 <- postResample(pred = preds, obs = obs)
  res_enet1_b3 <- c(res_enet1_b3, Balanced_Accuracy = balanced_acc, Sensitivity = sensitivity, Specificity = specificity)
  
  res1_b3 <- rbind(res1_b3, cbind('iter' = i, 'mod_b3' = 1, res_enet1_b3))
  
  
  #### Analyse für die Kombination der 3 Blöcke analog zu der in Block 1 ####
  
  # Training des Elastic-Net-Modells für Block 1, 2, 3
  enet1_b1_b2_b3 <- train(mod_b1_b2_b3,
                          data_final[, c(Präd_b1_b2_b3, "replicate")],
                          metric = "logLoss",
                          method = "glmnet",
                          family = "binomial",
                          trControl = train_control_loocv_sd,
                          tuneGrid = grid) 
  
  preds <- predict(enet1_b1_b2_b3, data_final)
  obs <- data_final$replicate
  
  cm <- caret::confusionMatrix(preds, obs)
  balanced_acc <- cm$byClass["Balanced Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # Abspeichern der Ergebnisse im Ergebnisobjekt 
  res_enet1_b1_b2_b3 <- postResample(pred = preds, obs = obs)
  res_enet1_b1_b2_b3 <- c(res_enet1_b1_b2_b3, Balanced_Accuracy = balanced_acc, Sensitivity = sensitivity, Specificity = specificity)
  
  res1_b1_b2_b3 <- rbind(res1_b1_b2_b3, cbind('iter' = i, 'mod_b1_b2_b3' = 1, res_enet1_b1_b2_b3))
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


for (i in 1:1) { # Outer Loop mit einer Iteration, um alle CART Modelle zu schätzen
  print(paste0(i, ". Iter"))
  
  set.seed(20190930 + i) # Seed gewährleistet Reproduzierbarkeit der Ergebnisse
  data_final <- na.omit(data_final) # Entfernen von NA-Werten aus den Daten
  
  train_control_loocv <- trainControl(method = "LOOCV", # Definieren der Leave-on-out Cross Validation
                                      summaryFunction = mnLogLoss, # standardisieren der Daten für jede Iteration der Cross Validation
                                      classProbs = TRUE,# Klassenwahrscheinlichkeiten 
                                      savePredictions = TRUE) # Vorhersagen speichern 
  
  # Grid Search wird bei Verfahren mit mehr als einem Hyperparameter vernwendet
  grid <- expand.grid(.cp = seq(.01, .10, .001)) # GridSearch um Hyperparameter zu optimieren
                                                 # verschiede Sequenzen getestet (siehe Anhang)
  # für minsplit, minbucket & maxdepth werden Defaults verwendet 
 
   #### Modelle ####
  
  #### Analyse Prädiktoren Block 1 ####
  
  # Modellspezifikation Block 1
  cart1_b1 <- train(mod_b1, # zuvor spezifiziertes Modell 
                    data = data_final, 
                    metric = "logLoss", # logLoss als Metrik zur Bewertung von Klassifikationsfragestellungen
                    method = "rpart", # Methode für Decision Trees
                    tuneGrid = grid, # zuvor spezifiziertes TuneGrid wählen
                    # tuneLength = 21  -> als Alternative für GridSearch genutzt, alternativ auch tuneLength = 50 (random search)
                    trControl = train_control_loocv) # Definieren der Art der Cross Validation (s.o.)
  
  preds <- predict(cart1_b1, data_final) # Definieren notwendiger Objekte für das
  obs <- data_final$replicate            # Erstellen der Confusion Matrix
  
  cm <- caret::confusionMatrix(preds, obs) # Erstellen einer Confusion Matrix zur Berechnung der Balanced Accuracy
  balanced_acc <- cm$byClass["Balanced Accuracy"]     # Ableiten von BACC...
  sensitivity <- cm$byClass["Sensitivity"]            # ... Senitivität und ...          
  specificity <- cm$byClass["Specificity"]            # ... Spezifität aus der ConfusionMatrix
 
  # Abspeichern der Ergebnisse im Ergebnisobjekt   
  res_cart1_b1 <- postResample(pred = preds, obs = obs)
  res_cart1_b1 <- c(res_cart1_b1, Balanced_Accuracy = balanced_acc, Sensitivity = sensitivity, Specificity = specificity) # Als Outputvariablen BACC, Sen und Spe definiert.
  
  res1_c_b1 <- rbind(res1_c_b1, cbind('iter' = i, 'mod_b1' = 2, res_cart1_b1)) # Ergebnisse im Ergebnisobjekt abspeichern
  
  #### Analyse für Block 2 analog zu der in Block 1 ####
  
  # Modellspezifikation Block 2
  
  cart1_b2 <- train(mod_b2,
                    data = data_final,
                    metric = "logLoss",
                    method = "rpart",
                    tuneGrid = grid,
                    trControl = train_control_loocv)
  
  preds <- predict(cart1_b2, data_final)
  obs <- data_final$replicate
  
  cm <- caret::confusionMatrix(preds, obs)
  balanced_acc <- cm$byClass["Balanced Accuracy"]   
  sensitivity <- cm$byClass["Sensitivity"]            
  specificity <- cm$byClass["Specificity"]   
  
  # Abspeichern der Ergebnisse im Ergebnisobjekt 
  res_cart1_b2 <- postResample(pred = preds, obs = obs)
  res_cart1_b2 <- c(res_cart1_b2, Balanced_Accuracy = balanced_acc, Sensitivity = sensitivity, Specificity = specificity)
  
  res1_c_b2 <- rbind(res1_c_b2, cbind('iter' = i, 'mod_b2' = 2, res_cart1_b2))
  
  #### Analyse für Block 3 analog zu der in Block 1 ####
  
  # Modellspezifikation Block 3
  
  cart1_b3 <- train(mod_b3,
                    data = data_final,
                    metric = "logLoss",
                    method = "rpart",
                    tuneGrid = grid,
                    trControl = train_control_loocv)
  
  preds <- predict(cart1_b3, data_final)
  obs <- data_final$replicate
  
  cm <- caret::confusionMatrix(preds, obs)
  balanced_acc <- cm$byClass["Balanced Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # Abspeichern der Ergebnisse im Ergebnisobjekt 
  res_cart1_b3 <- postResample(pred = preds, obs = obs)
  res_cart1_b3 <- c(res_cart1_b3, Balanced_Accuracy = balanced_acc, Sensitivity = sensitivity, Specificity = specificity)
  
  res1_c_b3 <- rbind(res1_c_b3, cbind('iter' = i, 'mod_b3' = 2, res_cart1_b3))
  
  #### Analyse für die Kombination der 3 Blöcke analog zu der in Block 1 ####
  
  # Modellspezifikation Block 1, 2, 3
  
  cart1_b1_b2_b3 <- train(mod_b1_b2_b3,
                          data = data_final,
                          metric = "logLoss",
                          method = "rpart",
                          tuneGrid = grid,
                          trControl = train_control_loocv)
  
  preds <- predict(cart1_b1_b2_b3, data_final)
  obs <- data_final$replicate
  
  cm <- caret::confusionMatrix(preds, obs)
  balanced_acc <- cm$byClass["Balanced Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  
  # Abspeichern der Ergebnisse im Ergebnisobjekt 
  res_cart1_b1_b2_b3 <- postResample(pred = preds, obs = obs)
  res_cart1_b1_b2_b3 <- c(res_cart1_b1_b2_b3, Balanced_Accuracy = balanced_acc, Sensitivity = sensitivity, Specificity = specificity)
  
  res1_c_b1_b2_b3 <- rbind(res1_c_b1_b2_b3, cbind('iter' = i, 'mod_b1_b2_b3' = 2, res_cart1_b1_b2_b3))
}



# ______________________________________________________________________________
#
#  Ergebnisse-ENet für jeden Block
#
# ______________________________________________________________________________


# Auswertung Prädiktorenblock 1

res1_b1
vip(enet1_b1, num_features = 20)
plot(enet1_b1)

coefficients_enet1_b1 <- coef(enet1_b1$finalModel, s = enet1_b1$bestTune$lambda)
print(coefficients_enet1_b1) # Koeffizienten

enet1_b1$bestTune # beste Kombination aus Hyperparametern



# Auswertung Prädiktorenblock 2

res1_b2
vip(enet1_b2, num_features = 20)
plot(enet1_b2)

coefficients_enet1_b2 <- coef(enet1_b2$finalModel, s = enet1_b2$bestTune$lambda)
print(coefficients_enet1_b2) # Koeffizienten

enet1_b2$bestTune # beste Kombination aus Hyperparametern




# Auswertung Prädiktorenblock 3

res1_b3
vip(enet1_b3, num_features = 20)
plot(enet1_b3)

coefficients_enet1_b3 <- coef(enet1_b3$finalModel, s = enet1_b3$bestTune$lambda)
print(coefficients_enet1_b3) # Koeffizienten

enet1_b3$bestTune # beste Kombination aus Hyperparametern



# Auswertung Prädiktorenblock 1,2 und 3

res1_b1_b2_b3
vip(enet1_b1_b2_b3, num_features = 20)
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

res1_c_b1
cbind(res1_b1, res1_c_b1)

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

res1_c_b2
cbind(res1_b2, res1_c_b2)

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

res1_c_b3
cbind(res1_b3, res1_c_b3)

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

res1_c_b1_b2_b3
cbind(res1_b1_b2_b3, res1_c_b1_b2_b3)

vip(cart1_b1_b2_b3, num_features = 20)
plot(cart1_b1_b2_b3)

cart1_b1_b2_b3$finalModel$cp
cart1_b1_b2_b3$finalModel$control$minsplit
cart1_b1_b2_b3$finalModel$control$minbucket
cart1_b1_b2_b3$finalModel$control$maxdepth

rpart.plot::rpart.plot(cart1_b1_b2_b3$finalModel,
                       fallen.leaves = TRUE,
                       box.palette = "RdGn")
