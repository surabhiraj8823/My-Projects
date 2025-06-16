import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { RegisterComponent } from './register/register.component';
import { LoginComponent } from './login/login.component';
import { CustomerComponent } from './customer/customer.component';
import { AuthGuard } from './guard/auth.guard';
import { SignupComponent } from './signup/signup.component';

const routes: Routes = [
  { path: 'customer', component: CustomerComponent, canActivate:[AuthGuard] },
  { path: 'register', component: RegisterComponent },          
  { path: 'register/:id', component: RegisterComponent },      
  { path: '', component: LoginComponent },
  { path: 'login', component: LoginComponent },
  { path: 'signup', component: SignupComponent }
];


@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
