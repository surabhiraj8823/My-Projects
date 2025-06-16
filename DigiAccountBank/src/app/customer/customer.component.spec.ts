import { TestBed } from '@angular/core/testing';
import { CustomerComponent } from './customer.component';
import { HttpClientModule } from '@angular/common/http'; 
import { LoginService } from '../services/login.service'; 

describe('CustomerComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HttpClientModule], 
      providers: [LoginService],  
    }).compileComponents();
  });

  it('should create', () => {
    const fixture = TestBed.createComponent(CustomerComponent);
    const component = fixture.componentInstance;
    expect(component).toBeTruthy();
  });
});