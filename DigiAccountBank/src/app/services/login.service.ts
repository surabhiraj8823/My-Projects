import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { Agents, User } from '../models/user';

@Injectable({
  providedIn: 'root'
})
export class LoginService {
  private customerUrl = 'http://localhost:3000/userdetails';  
  private countriesUrl = 'http://localhost:3000/countries';
  private agentsUrl = 'http://localhost:3000/agents';  


  constructor(private client: HttpClient) {}
  validate(email: string, password: string, recaptcha: any): Observable<boolean> {
    return this.client.get<Agents[]>(this.agentsUrl).pipe(
      map(agents => agents.some(agent => agent.email === email && agent.password === password))
    );
  }

  addAgent(agent: Agents): Observable<Agents> {
    return this.client.post<Agents>(this.agentsUrl, agent);
  }

  getAgents(): Observable<Agents[]> {
    return this.client.get<Agents[]>(this.agentsUrl);
  }


  addUserdetails(customer: User): Observable<User> {
    return this.client.post<User>(this.customerUrl, customer);
  }

  getCountries(): Observable<any> {
    return this.client.get(this.countriesUrl);
  }

  getUsers(): Observable<User[]> {
    return this.client.get<User[]>(this.customerUrl);
  }

  getUserById(id: string): Observable<User> {
    return this.client.get<User>(`${this.customerUrl}/${id}`);
  }

  deleteUser(id: string): Observable<void> {
    return this.client.delete<void>(`${this.customerUrl}/${id}`);
  }

  updateUser(id: string, updatedUser: User): Observable<User> {
    return this.client.put<User>(`${this.customerUrl}/${id}`, updatedUser);
  }
}
