#include "EventAction.hh"

#include "RunAction.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

namespace B1
{

EventAction::EventAction(RunAction* runAction)
: G4UserEventAction(),
  fRunAction(runAction),
  fDetectorEdep(0.),
  fDetectorGammaEntries(0),
  fPrimaryGammaEntries(0)
{}

void EventAction::BeginOfEventAction(const G4Event*)
{
  fDetectorEdep = 0.;
  fDetectorGammaEntries = 0;
  fPrimaryGammaEntries = 0;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  fRunAction->AddRunDetectorEdep(fDetectorEdep);
  fRunAction->AddRunDetectorGammaEntries(fDetectorGammaEntries);
  fRunAction->AddRunPrimaryGammaEntries(fPrimaryGammaEntries);

  fRunAction->WriteEventData(
    event->GetEventID(),
    fDetectorEdep / keV,
    fDetectorGammaEntries,
    fPrimaryGammaEntries);
}

void EventAction::RecordDetectorHit(G4int eventID,
                                    G4double y_mm,
                                    G4double z_mm,
                                    G4double photonEnergy_keV,
                                    G4bool isPrimary,
                                    G4double theta_deg,
                                    G4bool isDirectPrimary,
                                    G4bool isScatteredPrimary)
{
  fRunAction->WriteHitData(
    eventID,
    y_mm,
    z_mm,
    photonEnergy_keV,
    isPrimary,
    theta_deg,
    isDirectPrimary,
    isScatteredPrimary);
}

}