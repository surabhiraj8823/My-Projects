import { TestBed } from '@angular/core/testing';
import { AuthGuard } from './auth.guard';
import { RouterTestingModule } from '@angular/router/testing';
import { ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';

describe('AuthGuard', () => {
  let guard: AuthGuard;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      providers: [AuthGuard],
    });

    guard = TestBed.inject(AuthGuard);
  });

  it('should return true when isAuthenticated is true in localStorage', () => {
    spyOn(localStorage, 'getItem').and.callFake((key: string) => {
      if (key === 'status') return 'loggedIn'; // Mock the correct value for 'status'
      if (key === 'email') return 'test@example.com'; // Mock an email value
      return null;
    });

    const mockRoute = {} as ActivatedRouteSnapshot;
    const mockState = {} as RouterStateSnapshot;

    expect(guard.canActivate(mockRoute, mockState)).toBeTrue();
  });

  it('should return false when isAuthenticated is false in localStorage', () => {
    spyOn(localStorage, 'getItem').and.returnValue(null); // Mock no value for 'status'

    const mockRoute = {} as ActivatedRouteSnapshot;
    const mockState = {} as RouterStateSnapshot;

    expect(guard.canActivate(mockRoute, mockState)).toBeFalse();
  });
});