import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ContextExtractComponent } from './context-extract.component';

describe('ContextExtractComponent', () => {
  let component: ContextExtractComponent;
  let fixture: ComponentFixture<ContextExtractComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ContextExtractComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ContextExtractComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
