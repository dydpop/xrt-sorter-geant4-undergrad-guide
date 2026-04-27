#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "ExperimentConfig.hh"

#include "G4EmLivermorePhysics.hh"
#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "QBBC.hh"
#include "Randomize.hh"

int main(int argc, char** argv)
{
  const auto& config = GetExperimentConfig();
  if (config.randomSeed >= 0) {
    CLHEP::HepRandom::setTheSeed(config.randomSeed);
  }

  G4UIExecutive* ui = nullptr;
  if (argc == 1) {
    ui = new G4UIExecutive(argc, argv);
  }

  auto* runManager =
    G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);

  // Geometry
  runManager->SetUserInitialization(new DetectorConstruction());

  // Physics:
  // keep QBBC shell, but explicitly replace EM physics with Livermore
  auto* physicsList = new QBBC;
  physicsList->ReplacePhysics(new G4EmLivermorePhysics());
  physicsList->SetVerboseLevel(1);
  runManager->SetUserInitialization(physicsList);

  // User actions
  runManager->SetUserInitialization(new B1::ActionInitialization());

  // Visualization
  auto* visManager = new G4VisExecutive;
  visManager->Initialize();

  auto* UImanager = G4UImanager::GetUIpointer();

  if (!ui) {
    G4String command = "/control/execute ";
    G4String fileName = argv[1];
    UImanager->ApplyCommand(command + fileName);
  }
  else {
    UImanager->ApplyCommand("/control/execute init_vis.mac");
    ui->SessionStart();
    delete ui;
  }

  delete visManager;
  delete runManager;

  return 0;
}
