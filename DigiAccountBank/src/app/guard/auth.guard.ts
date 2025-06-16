import { Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  constructor(private router: Router) {}

  canActivate(route: unknown, state: unknown): boolean {
    const isAuthenticated = !!localStorage.getItem('status') && localStorage.getItem('status') === 'loggedIn';
    if (isAuthenticated) {
      const email = localStorage.getItem('email');
      if (email) {
        console.log('User is authenticated with email:', email);
      } else {
        console.error('Email not found in local storage.');
      }
    }
     else {
      console.log('User is not authenticated. Redirecting to login page.');
      this.router.navigate(['login']);
      alert('Please login to access the dashboard.');
    } 
    
    return isAuthenticated;
  }
}