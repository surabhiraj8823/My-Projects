import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ReactiveFormsModule } from '@angular/forms';
import { LoginComponent } from './login.component';
import { Router } from '@angular/router';
import { LoginService } from '../services/login.service';
import { of } from 'rxjs';
import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { NgxCaptchaModule } from 'ngx-captcha';

describe('LoginComponent', () => {
  let component: LoginComponent;
  let fixture: ComponentFixture<LoginComponent>;
  let mockRouter: jasmine.SpyObj<Router>;
  let mockLoginService: jasmine.SpyObj<LoginService>;

  beforeEach(async () => {
    mockRouter = jasmine.createSpyObj('Router', ['navigate']);
    mockLoginService = jasmine.createSpyObj('LoginService', ['validate']);

    await TestBed.configureTestingModule({
      declarations: [LoginComponent],
      imports: [
        ReactiveFormsModule,
        NgxCaptchaModule,
      ],
      providers: [
        { provide: Router, useValue: mockRouter },
        { provide: LoginService, useValue: mockLoginService },
      ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    }).compileComponents();

    fixture = TestBed.createComponent(LoginComponent);
    component = fixture.componentInstance;
    component.siteKey = '6LeoYUMrAAAAAAEIdurcQANa__eywuv3JDJeCAmA';

    // Ensure the reCAPTCHA element is properly initialized
    const recaptchaElement = document.createElement('div');
    recaptchaElement.id = 'ngx-recaptcha';
    document.body.appendChild(recaptchaElement);

    // Mock the reCAPTCHA element and its methods
    (component as any).captchaElem = {
      execute: jasmine.createSpy('execute').and.returnValue(of('valid-captcha')),
      reset: jasmine.createSpy('reset'),
    };

    // Ensure the reCAPTCHA element is properly referenced
    component.loginForm.patchValue({ recaptcha: 'valid-captcha' });
    fixture.detectChanges();
  });

  afterEach(() => {
    // Clean up the reCAPTCHA element after each test
    const recaptchaElement = document.getElementById('ngx-recaptcha');
    if (recaptchaElement) {
      document.body.removeChild(recaptchaElement);
    }
  });

  it('should create the component', () => {
    expect(component).toBeTruthy();
  });

  it('should have an invalid form when empty', () => {
    expect(component.loginForm.valid).toBeFalse();
  });

  it('should disable submit button if form is invalid', () => {
    const submitButton: HTMLButtonElement = fixture.nativeElement.querySelector('.submit');
    expect(submitButton.disabled).toBeTrue();
  });

  it('should enable submit button when form is valid', () => {
    component.loginForm.patchValue({
      email: 'agent@gmail.com',
      password: 'password123',
      recaptcha: 'valid-captcha',
    });
    fixture.detectChanges();
    const submitButton: HTMLButtonElement = fixture.nativeElement.querySelector('.submit');
    expect(submitButton.disabled).toBeFalse();
  });

  it('should call loginhandler on Sign In button click', () => {
    spyOn(component, 'loginhandler');
    component.loginForm.patchValue({
      email: 'agent@gmail.com',
      password: 'password123',
      recaptcha: 'valid-captcha',
    });
    fixture.detectChanges();
    const submitButton: HTMLButtonElement = fixture.nativeElement.querySelector('.submit');
    submitButton.click();
    expect(component.loginhandler).toHaveBeenCalled();
  });

  it('should call cancel on Cancel button click', () => {
    spyOn(component, 'cancel');
    const cancelButton: HTMLButtonElement = fixture.nativeElement.querySelector('.cancel');
    cancelButton.click();
    expect(component.cancel).toHaveBeenCalled();
  });

  it('should reset the form when cancel() is called', () => {
    component.loginForm.patchValue({
      email: 'agent@gmail.com',
      password: 'password123',
      recaptcha: 'valid-captcha',
    });
    component.cancel();
    expect(component.loginForm.value).toEqual({
      email: null,
      password: null,
      recaptcha: null,
    });
  });

  it('should invalidate password shorter than 6 characters', () => {
    component.loginForm.patchValue({ password: '12345' });
    expect(component.loginForm.get('password')?.valid).toBeFalse();
  });

  it('should validate password of at least 6 characters', () => {
    component.loginForm.patchValue({ password: '123456' });
    expect(component.loginForm.get('password')?.valid).toBeTrue();
  });

  it('should have valid email when format is correct', () => {
    component.loginForm.patchValue({ email: 'agent@gmail.com' });
    expect(component.loginForm.get('email')?.valid).toBeTrue();
  });

  it('should have email field invalid without @', () => {
    component.loginForm.patchValue({ email: 'invalidemail.com' });
    expect(component.loginForm.get('email')?.valid).toBeFalse();
  });
});