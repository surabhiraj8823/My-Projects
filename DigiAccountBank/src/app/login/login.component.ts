import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { LoginService } from '../services/login.service';

@Component({
  selector: 'app-login',
  standalone: false,
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  loginForm: FormGroup;

  constructor(private fb: FormBuilder, public us: LoginService, private router: Router) {
    this.loginForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6), Validators.maxLength(15)]],
      recaptcha: ['', Validators.required]
    });
  }

  siteKey: string = '6LeoYUMrAAAAAAEIdurcQANa__eywuv3JDJeCAmA';

  handleSuccess(e: any) {
    this.loginForm.patchValue({ recaptcha: e });
  }
  
  handleReset() {
    this.loginForm.patchValue({ recaptcha: '' });
  }

  handleExpire() {
    this.loginForm.patchValue({ recaptcha: '' });
  }

  loginhandler() {
    if (this.loginForm.invalid) {
      alert('Please fill in all required fields and complete the captcha.');
      return;
    }

    const { email, password, recaptcha } = this.loginForm.value;

    this.us.validate(email, password, recaptcha).subscribe(
      (isValid) => {
        if (isValid) {
          localStorage.setItem('status', 'loggedIn');
          localStorage.setItem('email', email);
          this.router.navigate(['customer']);
          alert('Logged In Successfully');
        } else {
          localStorage.setItem('status', 'notlogged');
          alert('Invalid email or password.');
        }
      },
      (error) => {
        console.error('Error during login:', error);
        alert('Captcha verification failed. Please try again.');
      }
    );
  }

  cancel() {
    this.loginForm.reset();
    window.location.reload();
  }
}