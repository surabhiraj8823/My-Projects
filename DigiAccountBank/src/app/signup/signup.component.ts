import { Component } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { LoginService } from '../services/login.service';
import { Agents } from '../models/user';
 
@Component({
  selector: 'app-signup',
  standalone: false,
  templateUrl: './signup.component.html',
  styleUrl: './signup.component.css'
})
export class SignupComponent {
  signupForm: FormGroup = new FormGroup({
    name: new FormControl(null, Validators.required),
    email: new FormControl(null, [Validators.required, Validators.email]),
    mobile: new FormControl(null, Validators.required),
    password: new FormControl(null, [Validators.minLength(6), Validators.maxLength(15), Validators.required]),
  });
 
  u: Agents = new Agents();
 
  constructor(public us: LoginService) {}
 
  handleSignup() {
    if (this.signupForm.valid) {
      const agent: Agents = this.signupForm.value;
      this.us.addAgent(agent).subscribe(() => {
        alert('Agent added successfully');
        this.signupForm.reset();
      });
    } else {
      alert('Please fill out the form correctly.');
    }
  }
 
  cancel() {
    this.signupForm.reset();
  }
}