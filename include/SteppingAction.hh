#ifndef B1SteppingAction_h
#define B1SteppingAction_h 1

#include "G4UserSteppingAction.hh"

class DetectorConstruction;
class G4LogicalVolume;
class G4Step;

namespace B1
{

class EventAction;

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction(const DetectorConstruction* detectorConstruction,
                   EventAction* eventAction);
    ~SteppingAction() override = default;

    void UserSteppingAction(const G4Step* step) override;

  private:
    const DetectorConstruction* fDetConstruction;
    EventAction* fEventAction;
    G4LogicalVolume* fScoringVolume;
};

}

#endif