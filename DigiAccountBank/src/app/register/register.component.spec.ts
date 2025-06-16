import { TestBed, ComponentFixture } from '@angular/core/testing';
import { ReactiveFormsModule } from '@angular/forms';
import { of } from 'rxjs';
import { ActivatedRoute, Router } from '@angular/router';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { RegisterComponent } from './register.component';
import { LoginService } from '../services/login.service';

describe('RegisterComponent', () => {
    let component: RegisterComponent;
    let fixture: ComponentFixture<RegisterComponent>;
    let mockLoginService: jasmine.SpyObj<LoginService>;
    let mockRouter: jasmine.SpyObj<Router>;
    let mockActivatedRoute: ActivatedRoute;

    beforeEach(async () => {
        mockLoginService = jasmine.createSpyObj('LoginService', [
            'addUserdetails',
            'getCountries',
            'getUserById',
            'updateUser'
        ]);
        mockRouter = jasmine.createSpyObj('Router', ['navigate']);
        mockActivatedRoute = { paramMap: of({ get: () => null }) } as unknown as ActivatedRoute;

        mockLoginService.getCountries.and.returnValue(
            of([
              {
                name: 'India',
                states: [
                  { name: 'Karnataka', cities: ['Bangalore', 'Mysore'] },
                  { name: 'Maharashtra', cities: ['Mumbai', 'Pune'] }
                ]
              },
              {
                name: 'USA',
                states: [
                  { name: 'California', cities: ['Los Angeles', 'San Francisco'] },
                  { name: 'Texas', cities: ['Houston', 'Austin'] }
                ]
              }
            ])
          );

        await TestBed.configureTestingModule({
            imports: [
                ReactiveFormsModule,
            ],
            providers: [
                provideHttpClientTesting(),
                provideRouter([]),
                RegisterComponent,
                { provide: LoginService, useValue: mockLoginService },
                { provide: Router, useValue: mockRouter },
                { provide: ActivatedRoute, useValue: mockActivatedRoute }
            ]
        }).compileComponents();

        fixture = TestBed.createComponent(RegisterComponent);
        component = fixture.componentInstance;

        fixture.detectChanges();
    });

    it('should create the component', () => {
        expect(component).toBeTruthy();
    });

    it('should load countries on initialization', () => {
        component.loadCountries();
        expect(mockLoginService.getCountries).toHaveBeenCalled();
        expect(component.countries.length).toBeGreaterThan(0);
    });

    it('should validate email format', () => {
        const emailControl = component.registerForm.get('email_id'); 
        emailControl?.setValue('invalid-email'); 
        expect(emailControl?.valid).toBeFalse();
      
        emailControl?.setValue('valid@example.com'); 
        expect(emailControl?.valid).toBeTrue();
      });

    it('should require first name, last name, and email', () => {
        const firstNameControl = component.registerForm.get('first_name');
        const lastNameControl = component.registerForm.get('last_name');
        const emailControl = component.registerForm.get('email_id');
      
        firstNameControl?.setValue('');
        lastNameControl?.setValue('');
        emailControl?.setValue('');
      
        expect(firstNameControl?.valid).toBeFalse();
        expect(lastNameControl?.valid).toBeFalse();
        expect(emailControl?.valid).toBeFalse();
      });

      it('should validate phone number format', () => {
        const phoneControl = component.registerForm.get('address.phone_number'); 
        phoneControl?.setValue('12345');
        expect(phoneControl?.valid).toBeFalse();
      
        phoneControl?.setValue('9876543210'); 
        expect(phoneControl?.valid).toBeTrue();
      });

      it('should validate country change updates states', () => {
        component.countries = [
          {
            name: 'India',
            states: [
              { name: 'Karnataka', cities: ['Bangalore', 'Mysore'] },
              { name: 'Maharashtra', cities: ['Mumbai', 'Pune'] }
            ]
          },
          {
            name: 'USA',
            states: [
              { name: 'California', cities: ['Los Angeles', 'San Francisco'] },
              { name: 'Texas', cities: ['Houston', 'Austin'] }
            ]
          }
        ];
        component.onCountryChange({ target: { value: 'India' } });
        expect(component.states.length).toBeGreaterThan(0);
        expect(component.states).toEqual([
          { name: 'Karnataka', cities: ['Bangalore', 'Mysore'] },
          { name: 'Maharashtra', cities: ['Mumbai', 'Pune'] }
        ]);
      });

    it('should validate state change updates cities', () => {
        component.states = [
          { name: 'Karnataka', cities: ['Bangalore', 'Mysore'] },
          { name: 'Maharashtra', cities: ['Mumbai', 'Pune'] }
        ];
        component.onStateChange({ target: { value: 'Karnataka' } });
        expect(component.cities).toEqual(['Bangalore', 'Mysore']);
      });

    it('should validate aadhar number format', () => {
        const aadharControl = component.registerForm.get('addressProof.aadhar_number');
        aadharControl?.setValue('12345');
        expect(aadharControl?.valid).toBeFalse();

        aadharControl?.setValue('123456789012');
        expect(aadharControl?.valid).toBeTrue();
    });
});
