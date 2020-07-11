import { TestBed } from '@angular/core/testing';

import { RoleGuardServiceService } from './role-guard-service.service';

describe('RoleGuardServiceService', () => {
  let service: RoleGuardServiceService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(RoleGuardServiceService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
